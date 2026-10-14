/*
 * SPDX-FileCopyrightText: 2026 KritaGPU contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include "KisGpuBrushPaint.h"
#include "KisGpuBrushCompositor.h"

#include <QDebug>
#include <QCoreApplication>
#include <QHash>
#include <QSet>
#include <QRegion>
#include <KoColorSpace.h>
#include <KoColorSpaceConstants.h>
#include <KoColorModelStandardIds.h>
#include <KoCompositeOpRegistry.h>
#include <compositeops/KoAlphaDarkenParamsWrapper.h>
#include <KisRenderedDab.h>
#include <kis_paint_device.h>
#include <kis_painter.h>
#include <kis_pixel_selection.h>
#include <kis_selection.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <new>
#include <mutex>

namespace KisGpuBrushPaint
{
namespace
{
constexpr std::size_t maxPixels = 32u * 1024u * 1024u;

bool supported(KisPainter *painter, const QList<KisRenderedDab> &dabs)
{
#ifndef Q_OS_WIN
    Q_UNUSED(painter);
    Q_UNUSED(dabs);
    return false;
#else
    if (!painter || !painter->device() || dabs.isEmpty() ||
        painter->device()->defaultBounds()->wrapAroundMode()) return false;
    const KoColorSpace *cs = painter->device()->colorSpace();
    if (cs->colorModelId() != RGBAColorModelID ||
        cs->colorDepthId() != Integer8BitsColorDepthID ||
        cs->pixelSize() != 4 || cs->alphaPos() != 3) return false;
    const QString op = painter->compositeOpId();
    if (op != COMPOSITE_OVER && op != COMPOSITE_ALPHA_DARKEN && op != COMPOSITE_ERASE) return false;
    if (!painter->channelFlags().isEmpty() && op != COMPOSITE_OVER) return false;
    if (!painter->channelFlags().isEmpty() && painter->channelFlags().size() != 4) return false;
    for (const auto &dab : dabs) {
        if (!dab.device || *dab.device->colorSpace() != *cs) return false;
    }
    return true;
#endif
}

bool buildBatch(KisPainter *painter, const QVector<QRect> &rects,
                const QList<KisRenderedDab> &dabs, KisGpuBrush::Batch &batch)
{
    using namespace KisGpuBrush;
    const QString op = painter->compositeOpId();
    batch.mode = op == COMPOSITE_OVER ? BlendMode::Over :
        op == COMPOSITE_ERASE ? BlendMode::Erase :
        useCreamyAlphaDarken() ? BlendMode::AlphaDarkenCreamy : BlendMode::AlphaDarkenHard;
    const QBitArray flags = painter->channelFlags();
    if (!flags.isEmpty()) {
        batch.channelFlags = 0;
        for (int i = 0; i < 4; ++i) if (flags.testBit(i)) batch.channelFlags |= 1u << i;
    }

    QHash<const KisFixedPaintDevice *, std::uint32_t> sourceOffsets;
    QRect allDabs;
    for (const KisRenderedDab &dab : dabs) {
        const QRect bounds = dab.realBounds();
        if (bounds.isEmpty()) return false;
        const std::uint64_t count = std::uint64_t(bounds.width()) * bounds.height();
        if (count > maxPixels || count > std::uint64_t(dab.device->allocatedPixels())) return false;
        const KisFixedPaintDevice *key = dab.device.data();
        if (!sourceOffsets.contains(key)) {
            if (batch.sourcePixels.size() + count > maxPixels) return false;
            const auto offset = std::uint32_t(batch.sourcePixels.size());
            batch.sourcePixels.resize(batch.sourcePixels.size() + std::size_t(count));
            std::memcpy(batch.sourcePixels.data() + offset, dab.device->constData(), std::size_t(count) * 4);
            sourceOffsets.insert(key, offset);
        }
        Dab gpuDab;
        gpuDab.bounds[0] = bounds.x(); gpuDab.bounds[1] = bounds.y();
        gpuDab.bounds[2] = bounds.width(); gpuDab.bounds[3] = bounds.height();
        gpuDab.data[0] = sourceOffsets.value(key);
        gpuDab.params[0] = float(dab.opacity);
        gpuDab.params[1] = float(dab.flow);
        gpuDab.params[2] = float(dab.averageOpacity);
        batch.dabs.push_back(gpuDab);
        allDabs |= bounds;
    }
    const KisSelectionSP selection = painter->selection();
    // The CPU scheduler splits dense work into small parallel patches. Merge
    // adjacent patches again for the GPU, preserving their exact union, so a
    // 512px area can use one dispatch instead of sixteen 128px dispatches.
    QRegion updateRegion;
    for (const QRect &rect : rects) updateRegion += rect;
    for (QRect rect : updateRegion) {
        rect &= allDabs;
        if (selection) rect &= selection->selectedRect();
        if (rect.isEmpty()) continue;
        // Use int64 loop bounds because QRect can extend to negative canvas
        // coordinates and its inclusive right/bottom can be INT_MAX.
        for (std::int64_t y = rect.y(); y < std::int64_t(rect.y()) + rect.height(); y += 512) {
            for (std::int64_t x = rect.x(); x < std::int64_t(rect.x()) + rect.width(); x += 512) {
                const QRect tileRect(int(x), int(y),
                    int(std::min<std::int64_t>(512, std::int64_t(rect.x()) + rect.width() - x)),
                    int(std::min<std::int64_t>(512, std::int64_t(rect.y()) + rect.height() - y)));
                Tile tile;
                tile.x = tileRect.x(); tile.y = tileRect.y();
                tile.width = tileRect.width(); tile.height = tileRect.height();
                tile.indexOffset = std::uint32_t(batch.dabIndices.size());
                for (int i = 0; i < dabs.size(); ++i) {
                    if (tileRect.intersects(dabs[i].realBounds())) batch.dabIndices.push_back(std::uint32_t(i));
                }
                tile.dabCount = std::uint32_t(batch.dabIndices.size()) - tile.indexOffset;
                if (!tile.dabCount) continue;
                if (tile.dabCount > 4096 || batch.dabIndices.size() > maxPixels) return false;
                const std::size_t count = std::size_t(tile.width) * tile.height;
                if (batch.pixels.size() + count > maxPixels) return false;
                tile.pixelOffset = std::uint32_t(batch.pixels.size());
                batch.pixels.resize(batch.pixels.size() + count);
                painter->device()->readBytes(reinterpret_cast<quint8 *>(batch.pixels.data() + tile.pixelOffset), tileRect);
                const auto oldSize = batch.selection.size();
                batch.selection.resize(oldSize + count, 255);
                if (selection) {
                    std::vector<quint8> mask(count);
                    selection->projection()->readBytes(mask.data(), tileRect);
                    std::copy(mask.begin(), mask.end(), batch.selection.begin() + oldSize);
                }
                batch.tiles.push_back(tile);
            }
        }
    }
    return true;
}
}

bool shouldUse(KisPainter *painter, const QVector<QRect> &rects, const QList<KisRenderedDab> &dabs)
{
    if (!supported(painter, dabs)) return false;
    if (dabs.size() > 4096) return false;
    QSet<const KisFixedPaintDevice *> uniqueTips;
    std::uint64_t sourcePixels = 0, destinationPixels = 0;
    for (const auto &dab : dabs) {
        if (!uniqueTips.contains(dab.device.data())) {
            uniqueTips.insert(dab.device.data());
            const QRect bounds = dab.device->bounds();
            if (bounds.isEmpty()) return false;
            sourcePixels += std::uint64_t(bounds.width()) * bounds.height();
            if (sourcePixels > maxPixels) return false;
        }
    }
    for (const auto &rect : rects) {
        if (rect.isEmpty()) continue;
        destinationPixels += std::uint64_t(rect.width()) * rect.height();
        if (destinationPixels > maxPixels) return false;
    }
    // Transfers dominate small dabs. Dispatch only sizeable overlapping work;
    // the override is useful for correctness tests and RenderDoc captures.
    if (!qEnvironmentVariableIsSet("KRITA_GPU_BRUSH_FORCE")) {
        if (dabs.size() < 4) return false;
        std::uint64_t work = 0;
        for (const auto &dab : dabs) {
            for (const QRect &rect : rects) {
                const QRect overlap = rect & dab.realBounds();
                work += std::uint64_t(std::max(0, overlap.width())) * std::max(0, overlap.height());
            }
        }
        if (work < 4u * 1024u * 1024u) return false;
    }
    // Finish the WGL worker before plugin DLL teardown. Joining a worker from
    // a DLL's process-detach callback can otherwise deadlock the Windows loader.
    static std::once_flag shutdownConnection;
    std::call_once(shutdownConnection, [] {
        if (auto *app = QCoreApplication::instance()) {
            QObject::connect(app, &QCoreApplication::aboutToQuit, app, [] {
                KisGpuBrush::Compositor::instance().shutdown();
            }, Qt::DirectConnection);
        }
    });
    const bool available = KisGpuBrush::Compositor::instance().isAvailable();
    static std::atomic<bool> reported {false};
    if (!reported.exchange(true)) qInfo().noquote() << QString::fromStdString(KisGpuBrush::Compositor::instance().diagnostic());
    return available;
}

bool tryPaint(KisPainter *painter, const QVector<QRect> &rects, const QList<KisRenderedDab> &dabs)
{
    if (!supported(painter, dabs)) return false;
    KisGpuBrush::Batch batch;
    KisGpuBrush::Result result;
    try {
        if (!buildBatch(painter, rects, dabs, batch)) return false;
        if (batch.tiles.empty()) return true;
        result = KisGpuBrush::Compositor::instance().render(batch);
        if (!result.success) {
            qWarning().noquote() << "GPU brush batch fell back to CPU:" << QString::fromStdString(result.message);
            return false;
        }
    } catch (const std::bad_alloc &) {
        return false;
    }
    // The stroke's transaction owns these writes, just as it owns CPU blits.
    // Once commit begins, an allocation exception must propagate like it does
    // in the CPU painter; retrying after a partial write would paint twice.
    for (const auto &tile : batch.tiles) {
        painter->device()->writeBytes(reinterpret_cast<const quint8 *>(batch.pixels.data() + tile.pixelOffset),
                                     QRect(tile.x, tile.y, tile.width, tile.height));
    }
    if (qEnvironmentVariableIsSet("KRITA_GPU_BRUSH_LOG")) {
        qInfo() << "GPU brush batch:" << dabs.size() << "dabs," << result.dispatches << "compute dispatches";
    }
    return true;
}
}
