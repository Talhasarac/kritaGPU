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
std::mutex activeSessionMutex;
QHash<KisPaintDevice *,Session *> activeSessions;

bool pixelFormat(const KoColorSpace *cs, KisGpuBrush::PixelFormat *format)
{
    using F = KisGpuBrush::PixelFormat;
    if (cs->colorModelId() != RGBAColorModelID) return false;
    if (cs->colorDepthId() == Integer8BitsColorDepthID) *format = F::RGBA8;
    else if (cs->colorDepthId() == Integer16BitsColorDepthID) *format = F::RGBA16;
    else if (cs->colorDepthId() == Float16BitsColorDepthID) *format = F::RGBA16F;
    else if (cs->colorDepthId() == Float32BitsColorDepthID) *format = F::RGBA32F;
    else return false;
    return cs->pixelSize() == 4 * KisGpuBrush::wordsPerPixel(*format) && cs->alphaPos() == 3;
}

bool blendMode(const QString &op, KisGpuBrush::BlendMode *mode)
{
    using M = KisGpuBrush::BlendMode;
    static const std::pair<QString, M> modes[] = {
        {COMPOSITE_OVER,M::Over}, {COMPOSITE_ERASE,M::Erase},
        {COMPOSITE_MULT,M::Multiply}, {COMPOSITE_SCREEN,M::Screen}, {COMPOSITE_OVERLAY,M::Overlay},
        {COMPOSITE_DARKEN,M::Darken}, {COMPOSITE_LIGHTEN,M::Lighten}, {COMPOSITE_ADD,M::Add},
        {COMPOSITE_SUBTRACT,M::Subtract}, {COMPOSITE_COPY,M::Copy}, {COMPOSITE_DIFF,M::Difference},
        {COMPOSITE_EXCLUSION,M::Exclusion}, {COMPOSITE_DODGE,M::ColorDodge}, {COMPOSITE_BURN,M::ColorBurn},
        {COMPOSITE_HARD_LIGHT,M::HardLight}, {COMPOSITE_SOFT_LIGHT_SVG,M::SoftLight}, {COMPOSITE_LINEAR_BURN,M::LinearBurn}
    };
    if (op == COMPOSITE_ALPHA_DARKEN) { *mode = useCreamyAlphaDarken() ? M::AlphaDarkenCreamy : M::AlphaDarkenHard; return true; }
    for (const auto &pair : modes) if (pair.first == op) { *mode = pair.second; return true; }
    return false;
}

bool supportedPainter(KisPainter *painter)
{
#ifndef Q_OS_WIN
    Q_UNUSED(painter);
    return false;
#else
    if (!painter || !painter->device() ||
        painter->device()->defaultBounds()->wrapAroundMode()) return false;
    const KoColorSpace *cs = painter->device()->colorSpace();
    KisGpuBrush::PixelFormat format;
    if (!pixelFormat(cs, &format)) return false;
    const QString op = painter->compositeOpId();
    KisGpuBrush::BlendMode mode;
    if (!blendMode(op, &mode)) return false;
    if (!painter->channelFlags().isEmpty() && (op == COMPOSITE_ALPHA_DARKEN || op == COMPOSITE_ERASE)) return false;
    if (!painter->channelFlags().isEmpty() && painter->channelFlags().size() != 4) return false;
    return true;
#endif
}

bool supported(KisPainter *painter, const QList<KisRenderedDab> &dabs)
{
    if (!supportedPainter(painter) || dabs.isEmpty()) return false;
    for (const auto &dab : dabs) {
        if (!dab.device || *dab.device->colorSpace() != *painter->device()->colorSpace()) return false;
    }
    return true;
}

bool buildBatch(KisPainter *painter, const QVector<QRect> &rects,
                const QList<KisRenderedDab> &dabs, KisGpuBrush::Batch &batch, Session *session)
{
    using namespace KisGpuBrush;
    if (!pixelFormat(painter->device()->colorSpace(), &batch.format)) return false;
    const auto words = wordsPerPixel(batch.format);
    batch.deferReadback=session && session->canDefer();
    if (batch.deferReadback) batch.presentationKey=session->presentationKey();
    if (!qEnvironmentVariableIsSet("KRITA_GPU_BRUSH_NO_RESIDENCY")) {
        if (session) {
            session->prepare(painter->device().data());
            batch.cacheId = session->id();
            batch.cacheGeneration = session->generation();
            const KisPaintDeviceSP device = painter->device();
            batch.loadPixels = [device](const Tile &tile, std::uint32_t *pixels) {
                device->readBytes(reinterpret_cast<quint8 *>(pixels),
                                  QRect(tile.x, tile.y, tile.width, tile.height));
            };
        } else {
            batch.cacheId = std::uint64_t(reinterpret_cast<std::uintptr_t>(painter->device().data()));
        }
    }
    const QString op = painter->compositeOpId();
    if (!blendMode(op, &batch.mode)) return false;
    const QBitArray flags = painter->channelFlags();
    if (!flags.isEmpty()) {
        batch.channelFlags = 0;
        for (int i = 0; i < 4; ++i) if (flags.testBit(i)) batch.channelFlags |= 1u << i;
    }

    QHash<const KisFixedPaintDevice *, std::uint32_t> sourceOffsets;
    QHash<const void *,std::uint32_t> assetOffsets;
    const auto appendAsset = [&](const std::shared_ptr<const std::vector<std::uint32_t>> &asset) -> std::uint32_t {
        if (!asset) return 0;
        const auto key = asset.get();
        if (assetOffsets.contains(key)) return assetOffsets.value(key);
        while (batch.sourcePixels.size() % words) batch.sourcePixels.push_back(0);
        if (batch.sourcePixels.size()+asset->size() > maxPixels) throw std::bad_alloc();
        const auto offset = std::uint32_t(batch.sourcePixels.size());
        batch.sourcePixels.insert(batch.sourcePixels.end(),asset->begin(),asset->end());
        assetOffsets.insert(key,offset);
        return offset;
    };
    QRect allDabs;
    for (const KisRenderedDab &dab : dabs) {
        const QRect bounds = dab.realBounds();
        if (bounds.isEmpty()) return false;
        const std::uint64_t count = std::uint64_t(bounds.width()) * bounds.height();
        if (count > maxPixels / words || count > std::uint64_t(dab.device->allocatedPixels())) return false;
        const KisFixedPaintDevice *key = dab.device.data();
        const auto generated = dab.device->deferredGpuDab();
        if (!generated && !sourceOffsets.contains(key)) {
            while (batch.sourcePixels.size() % words) batch.sourcePixels.push_back(0);
            if (batch.sourcePixels.size() + count * words > maxPixels) return false;
            const auto offset = std::uint32_t(batch.sourcePixels.size());
            batch.sourcePixels.resize(batch.sourcePixels.size() + std::size_t(count) * words);
            std::memcpy(batch.sourcePixels.data() + offset, dab.device->constData(), std::size_t(count) * words * 4);
            sourceOffsets.insert(key, offset);
        }
        Dab gpuDab = generated ? generated->description : Dab();
        gpuDab.bounds[0] = bounds.x(); gpuDab.bounds[1] = bounds.y();
        gpuDab.bounds[2] = bounds.width(); gpuDab.bounds[3] = bounds.height();
        if (generated) {
            gpuDab.data[0] = appendAsset(generated->source);
            if (gpuDab.data[1] == 5) {
                gpuDab.texture[0] += gpuDab.data[0];
                gpuDab.texture[1] += gpuDab.data[0];
            }
            if (generated->pattern) gpuDab.texture[0] = appendAsset(generated->pattern);
        } else gpuDab.data[0] = sourceOffsets.value(key);
        gpuDab.params[0] = float(dab.opacity);
        gpuDab.params[1] = float(dab.flow);
        gpuDab.params[2] = float(dab.averageOpacity);
        batch.dabs.push_back(gpuDab);
        allDabs |= bounds;
    }
    while (batch.sourcePixels.empty() || batch.sourcePixels.size() % words) batch.sourcePixels.push_back(0);
    const KisSelectionSP selection = painter->selection();
    // The CPU scheduler splits dense work into small parallel patches. Merge
    // adjacent patches again for the GPU, preserving their exact union, before
    // splitting at the stable resident-cell boundaries.
    QRegion updateRegion;
    for (const QRect &rect : rects) updateRegion += rect;
    for (QRect rect : updateRegion) {
        rect &= allDabs;
        if (selection) rect &= selection->selectedRect();
        if (rect.isEmpty()) continue;
        // Stable world-aligned cells survive moving update rectangles. Only
        // the dirty intersection is read back and written to the CPU device.
        // Cells are initialized in full only on cache misses. Use int64 bounds
        // because canvas coordinates can be negative or reach INT_MAX.
        const auto cellOrigin = [](int v) -> std::int64_t {
            return (std::int64_t(v) - (v < 0 ? residentTileSize - 1 : 0)) / residentTileSize * residentTileSize;
        };
        for (std::int64_t y = cellOrigin(rect.y()); y < std::int64_t(rect.y()) + rect.height(); y += residentTileSize) {
            for (std::int64_t x = cellOrigin(rect.x()); x < std::int64_t(rect.x()) + rect.width(); x += residentTileSize) {
                const QRect tileRect = QRect(int(x), int(y), residentTileSize, residentTileSize) & rect;
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
                if (batch.pixels.size() + count * words > maxPixels) return false;
                tile.pixelOffset = std::uint32_t(batch.pixels.size());
                batch.pixels.resize(batch.pixels.size() + count * words);
                if (!batch.loadPixels) {
                    painter->device()->readBytes(reinterpret_cast<quint8 *>(batch.pixels.data() + tile.pixelOffset), tileRect);
                }
                if (selection) {
                    const auto oldSize = batch.selection.size();
                    batch.selection.resize(oldSize + count);
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

} // anonymous namespace

bool trySmudge(KisPainter *painter, const QRect &rect, KisFixedPaintDevice *background,
               KisFixedPaintDevice *carried, KisFixedPaintDevice *mask, const KoColor &dulling,
               const KoColor &paint, bool useDulling, bool fused, const QString &smearOp,
               qreal smearOpacity, const QString &colourOp, qreal colourOpacity, Session *session)
{
    if (!supportedPainter(painter) || mask->pixelSize() != 1 ||
        rect.size() != mask->bounds().size() || rect.size() != background->bounds().size() ||
        (smearOp != COMPOSITE_OVER && smearOp != COMPOSITE_COPY) ||
        (colourOp != COMPOSITE_OVER && colourOp != COMPOSITE_COPY) ||
        *background->colorSpace() != *painter->device()->colorSpace()) return false;
    const std::size_t area = std::size_t(rect.width())*rect.height();
    if (area < 65536 && !qEnvironmentVariableIsSet("KRITA_GPU_BRUSH_FORCE")) return false;
    const auto words = background->pixelSize()/4;
    if (!words || area*(2*words+1) > maxPixels) return false;
    auto pixels = std::make_shared<std::vector<std::uint32_t>>(area*(2*words+1));
    std::memcpy(pixels->data(),useDulling ? background->constData() : carried->constData(),area*words*4);
    std::memcpy(pixels->data()+area*words,background->constData(),area*words*4);
    const quint8 *alpha = mask->constData();
    for (std::size_t i=0;i<area;++i) (*pixels)[area*2*words+i] = alpha[i];
    auto deferred = std::make_shared<KisDeferredGpuDab>();
    deferred->source = pixels;
    auto &d = deferred->description;
    d.data[1]=5; d.data[3]=fused ? 1 : 0;
    d.params[3]=useDulling ? 1 : 0;
    d.shape[0]=smearOpacity; d.shape[1]=smearOp==COMPOSITE_COPY ? 1 : 0;
    d.shape[2]=colourOpacity; d.shape[3]=colourOp==COMPOSITE_COPY ? 1 : 0;
    d.texture[0]=area*words; d.texture[1]=area*2*words;
    QVector<float> channels(4);
    paint.colorSpace()->normalisedChannelsValue(paint.data(),channels);
    for (int i=0;i<4;++i) d.colour[i]=channels[i]*255.0f;
    dulling.colorSpace()->normalisedChannelsValue(dulling.data(),channels);
    for (int i=0;i<4;++i) d.transform[i]=channels[i]*255.0f;
    KisFixedPaintDeviceSP tip = new KisFixedPaintDevice(background->colorSpace());
    tip->setRect(QRect(QPoint(),rect.size()));
    tip->setDeferredGpuDab(deferred);
    KisRenderedDab dab(tip);
    dab.offset=rect.topLeft(); dab.opacity=painter->opacityF();
    return tryPaint(painter,{rect},{dab},session);
}

struct Session::Pending {
    struct Entry { QVector<QRect> rects; QList<KisRenderedDab> dabs; };
    std::uint64_t key=0;
    KisPainter *painter=nullptr;
    KisPaintDeviceSP device;
    QVector<Entry> journal;
    QRegion dirty;
    QSet<const void *> assets;
    std::size_t bytes=0;
};

Session::Session()
    : m_pending(new Pending)
    , m_id([] { static std::atomic<std::uint64_t> next {1}; return next.fetch_add(1); }())
{
}

Session::~Session()
{
    // Normal finish/cancel callbacks flush or discard before the transaction
    // closes. Retain a recovery path for other paintop callers as well.
    try { flush(); } catch (const std::exception &e) { qCritical()<<"GPU stroke checkpoint:"<<e.what(); }
    std::lock_guard<std::mutex> lock(activeSessionMutex);
    for (auto it=activeSessions.begin();it!=activeSessions.end();) {
        if (it.value()==this) it=activeSessions.erase(it); else ++it;
    }
}

void Session::setPresentationKey(std::uint64_t key) { m_pending->key=key; }
std::uint64_t Session::presentationKey() const { return m_pending->key; }
bool Session::hasPending() const { return !m_pending->journal.isEmpty(); }
bool Session::canDefer() const
{
    return m_pending->key && !qEnvironmentVariableIsSet("KRITA_GPU_BRUSH_NO_RESIDENCY") &&
        KisGpuBrush::Compositor::instance().canPresent(m_pending->key);
}
void Session::enqueue(KisPainter *painter,const QVector<QRect> &rects,const QList<KisRenderedDab> &dabs)
{
    m_pending->painter=painter; m_pending->device=painter->device();
    Pending::Entry entry; entry.rects=rects;
    for (const auto &dab:dabs) {
        auto copy=dab;
        copy.device=new KisFixedPaintDevice(*dab.device);
        if (const auto generated=copy.device->deferredGpuDab()) {
            m_pending->bytes+=1024; // descriptor, snapshot and fixed-device overhead
            for (const auto &asset:{generated->source,generated->pattern}) {
                if (asset && !m_pending->assets.contains(asset.get())) {
                    m_pending->assets.insert(asset.get());
                    m_pending->bytes+=asset->size()*4;
                }
            }
        } else m_pending->bytes+=std::size_t(copy.device->bounds().width())*
            copy.device->bounds().height()*copy.device->pixelSize();
        entry.dabs.append(copy);
    }
    m_pending->journal.append(std::move(entry));
    for (const auto &rect:rects) m_pending->dirty+=rect;
    { std::lock_guard<std::mutex> lock(activeSessionMutex); activeSessions[painter->device().data()]=this; }
    committed(painter->device().data());
    if (m_pending->bytes>64u*1024u*1024u || m_pending->journal.size()>=256) flush();
}
void Session::flush()
{
    if (!hasPending()) return;
    auto &gpu=KisGpuBrush::Compositor::instance();
    auto checkpoint=gpu.isAvailable() ? gpu.checkpoint(m_id,m_generation) : KisGpuBrush::Checkpoint();
    if (checkpoint.success && !checkpoint.tiles.empty()) {
        for (const auto &tile:checkpoint.tiles) {
            m_pending->device->writeBytes(reinterpret_cast<const quint8 *>(checkpoint.pixels.data()+tile.pixelOffset),
                QRect(tile.x,tile.y,tile.width,tile.height));
        }
        if (qEnvironmentVariableIsSet("KRITA_GPU_BRUSH_LOG")) qInfo()<<"GPU stroke checkpoint:"
            <<checkpoint.pixels.size()*4<<"bytes read back for"<<m_pending->journal.size()<<"updates";
    } else {
        // Device loss can destroy a GPU-only update. Replay retained dabs from
        // the last CPU checkpoint inside the same undo transaction.
        for (const auto &entry:m_pending->journal) for (const auto &rect:entry.rects)
            m_pending->painter->bltFixed(rect,entry.dabs);
        qWarning()<<"GPU stroke replayed on CPU after checkpoint failure";
    }
    gpu.discard(m_id);
    QVector<QRect> dirty;
    for (const auto &rect:m_pending->dirty) dirty.append(rect);
    m_pending->device->setDirty(dirty);
    committed(m_pending->device.data());
    m_pending->journal.clear(); m_pending->dirty=QRegion(); m_pending->assets.clear(); m_pending->bytes=0;
}
void Session::discard()
{
    KisGpuBrush::Compositor::instance().discard(m_id);
    if (m_pending->device) m_pending->device->setDirty(m_pending->dirty.boundingRect());
    m_pending->journal.clear(); m_pending->dirty=QRegion(); m_pending->assets.clear(); m_pending->bytes=0;
    invalidate();
}
bool hasDeferred(KisPaintDevice *device)
{
    std::lock_guard<std::mutex> lock(activeSessionMutex);
    auto *session=activeSessions.value(device);
    return session && session->hasPending();
}
void flushDevice(KisPaintDevice *device)
{
    // The stroke scheduler keeps paintop lifetime stable across this call.
    Session *session;
    { std::lock_guard<std::mutex> lock(activeSessionMutex); session=activeSessions.value(device); }
    if (session) session->flush();
}
void discardDevice(KisPaintDevice *device)
{
    Session *session;
    { std::lock_guard<std::mutex> lock(activeSessionMutex); session=activeSessions.value(device); }
    if (session) session->discard();
}

void Session::prepare(KisPaintDevice *device)
{
    if (m_device != device || m_sequence != device->sequenceNumber() ||
        m_lod != device->defaultBounds()->currentLevelOfDetail() ||
        m_time != device->defaultBounds()->currentTime()) { flush(); ++m_generation; }
}

void Session::committed(KisPaintDevice *device)
{
    m_device = device;
    m_sequence = device->sequenceNumber();
    m_lod = device->defaultBounds()->currentLevelOfDetail();
    m_time = device->defaultBounds()->currentTime();
}

int maxDabsPerUpdate(KisPainter *painter, int averageDiameter)
{
    if (!supportedPainter(painter)) return -1;
    const auto diameter = std::uint64_t(std::max(1, averageDiameter));
    return int(std::clamp<std::uint64_t>((8u * 1024u * 1024u) / (diameter * diameter * painter->device()->pixelSize() / 4), 4, 256));
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
            sourcePixels += std::uint64_t(bounds.width()) * bounds.height() * painter->device()->pixelSize() / 4;
            if (const auto generated = dab.device->deferredGpuDab()) {
                sourcePixels -= std::uint64_t(bounds.width()) * bounds.height() * painter->device()->pixelSize() / 4;
                if (generated->source) sourcePixels += generated->source->size();
                if (generated->pattern) sourcePixels += generated->pattern->size();
            }
            if (sourcePixels > maxPixels) return false;
        }
    }
    for (const auto &rect : rects) {
        if (rect.isEmpty()) continue;
        destinationPixels += std::uint64_t(rect.width()) * rect.height() * painter->device()->pixelSize() / 4;
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

bool tryPaint(KisPainter *painter, const QVector<QRect> &rects, const QList<KisRenderedDab> &dabs,
              Session *session)
{
    if (session && session->hasPending() && !session->canDefer()) session->flush();
    if (!supported(painter, dabs)) { if (session) session->flush(); return false; }
    KisGpuBrush::Batch batch;
    KisGpuBrush::Result result;
    try {
        if (!buildBatch(painter, rects, dabs, batch, session)) { if (session) session->flush(); return false; }
        if (batch.tiles.empty()) return true;
        result = KisGpuBrush::Compositor::instance().render(batch);
        if (!result.success && session && session->hasPending() &&
            KisGpuBrush::Compositor::instance().isAvailable()) {
            session->flush();
            batch.deferReadback=session->canDefer();
            result=KisGpuBrush::Compositor::instance().render(batch);
        }
        if (!result.success) {
            if (session) session->flush();
            qWarning().noquote() << "GPU brush batch fell back to CPU:" << QString::fromStdString(result.message);
            return false;
        }
    } catch (const std::bad_alloc &) {
        if (session) session->flush();
        return false;
    }
    if (batch.deferReadback) {
        session->enqueue(painter,rects,dabs);
        if (qEnvironmentVariableIsSet("KRITA_GPU_BRUSH_LOG")) qInfo()<<"GPU direct brush batch:"
            <<dabs.size()<<"dabs,"<<result.uploadedBytes<<"uploaded bytes, 0 readback bytes";
        return true;
    }
    if (session && session->hasPending()) session->flush();
    // The stroke's transaction owns these writes, just as it owns CPU blits.
    // Once commit begins, an allocation exception must propagate like it does
    // in the CPU painter; retrying after a partial write would paint twice.
    for (const auto &tile : batch.tiles) {
        painter->device()->writeBytes(reinterpret_cast<const quint8 *>(batch.pixels.data() + tile.pixelOffset),
                                     QRect(tile.x, tile.y, tile.width, tile.height));
    }
    if (session) {
        if (batch.loadPixels) session->committed(painter->device().data());
        else session->invalidate();
    }
    if (qEnvironmentVariableIsSet("KRITA_GPU_BRUSH_LOG")) {
        qInfo() << "GPU brush batch:" << dabs.size() << "dabs," << result.dispatches << "compute dispatches,"
                << result.reusedTiles << "/" << result.residentTiles << "resident tile hits,"
                << result.destinationUploadedBytes << "destination bytes uploaded,"
                << result.uploadedBytes << "total bytes uploaded,"
                << result.readbackBytes << "bytes read back";
    }
    return true;
}
}
