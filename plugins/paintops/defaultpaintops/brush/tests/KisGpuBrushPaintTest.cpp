/*
 * SPDX-FileCopyrightText: 2026 KritaGPU contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#include "KisGpuBrushPaintTest.h"
#include <simpletest.h>
#include "../gpu/KisGpuBrushPaint.h"
#include "../gpu/KisGpuBrushCompositor.h"
#include <KoColorSpaceRegistry.h>
#include <KoCompositeOpRegistry.h>
#include <KisRenderedDab.h>
#include <kis_paint_device.h>
#include <kis_painter.h>
#include <kis_pixel_selection.h>
#include <kis_selection.h>
#include <kis_surrogate_undo_adapter.h>
#include <kis_transaction.h>
#include <QElapsedTimer>
#include <cmath>
#include <random>

namespace
{
const QRect testRect(-35, -17, 590, 535);
QByteArray readPixels(KisPaintDeviceSP device)
{
    QByteArray bytes(testRect.width() * testRect.height() * 4, Qt::Uninitialized);
    device->readBytes(reinterpret_cast<quint8 *>(bytes.data()), testRect);
    return bytes;
}
QList<KisRenderedDab> makeDabs()
{
    const KoColorSpace *cs = KoColorSpaceRegistry::instance()->rgb8();
    std::mt19937 random(54173);
    QList<KisRenderedDab> dabs;
    for (int i = 0; i < 15; ++i) {
        KisFixedPaintDeviceSP device;
        if (i % 3 == 0 && !dabs.isEmpty()) device = dabs.last().device;
        else {
            device = new KisFixedPaintDevice(cs);
            // Nonzero fixed-device origin must not become a source byte offset.
            device->setRect(QRect(12, 9, 550, 520));
            device->initialize();
            auto *p = device->data();
            for (int pixel = 0; pixel < 550 * 520; ++pixel) {
                for (int c = 0; c < 4; ++c) *p++ = quint8(1 + random() % 255);
            }
        }
        KisRenderedDab dab(device);
        dab.offset = QPoint(-30 + i * 4, -15 + i * 3);
        dab.opacity = 0.05 + 0.9 * (i % 7) / 7;
        dab.flow = i % 4 == 0 ? 1 : 0.2 + 0.7 * (i % 5) / 5;
        dab.averageOpacity = 0.1 + 0.8 * (i % 9) / 9;
        dabs.append(dab);
    }
    return dabs;
}
KisPaintDeviceSP makeDevice()
{
    KisPaintDeviceSP device = new KisPaintDevice(KoColorSpaceRegistry::instance()->rgb8());
    std::mt19937 random(1233);
    QByteArray bytes(testRect.width() * testRect.height() * 4, Qt::Uninitialized);
    for (int i = 0; i < bytes.size(); ++i) bytes[i] = char(random() % 256);
    for (int i = 3; i < bytes.size(); i += 20) bytes[i] = 0;
    device->writeBytes(reinterpret_cast<const quint8 *>(bytes.constData()), testRect);
    return device;
}
}

void KisGpuBrushPaintTest::initTestCase()
{
    if (!KisGpuBrush::Compositor::instance().isAvailable()) {
        QSKIP(KisGpuBrush::Compositor::instance().diagnostic().c_str());
    }
    qInfo().noquote() << QString::fromStdString(KisGpuBrush::Compositor::instance().diagnostic());
}

void KisGpuBrushPaintTest::testAgainstPainter_data()
{
    QTest::addColumn<QString>("op");
    QTest::addColumn<bool>("selected");
    QTest::addColumn<bool>("alphaLocked");
    QTest::newRow("normal") << QString(COMPOSITE_OVER) << false << false;
    QTest::newRow("normal-selection") << QString(COMPOSITE_OVER) << true << false;
    QTest::newRow("normal-alpha-lock") << QString(COMPOSITE_OVER) << true << true;
    QTest::newRow("wash") << QString(COMPOSITE_ALPHA_DARKEN) << false << false;
    QTest::newRow("wash-selection") << QString(COMPOSITE_ALPHA_DARKEN) << true << false;
    QTest::newRow("eraser") << QString(COMPOSITE_ERASE) << true << false;
}

void KisGpuBrushPaintTest::testAgainstPainter()
{
    QFETCH(QString, op);
    QFETCH(bool, selected);
    QFETCH(bool, alphaLocked);
    auto actual = makeDevice();
    KisPaintDeviceSP expected = new KisPaintDevice(*actual);
    KisPainter cpu(expected), gpu(actual);
    cpu.setCompositeOpId(op);
    gpu.setCompositeOpId(op);
    if (selected) {
        KisSelectionSP selection = new KisSelection();
        selection->pixelSelection()->select(QRect(-10, 5, 520, 470), 131);
        selection->pixelSelection()->select(QRect(15, 15, 70, 80), 255);
        cpu.setSelection(selection);
        gpu.setSelection(selection);
    }
    if (alphaLocked) {
        QBitArray flags(4, true);
        flags.clearBit(3);
        cpu.setChannelFlags(flags);
        gpu.setChannelFlags(flags);
    }
    const auto dabs = makeDabs();
    // Multiple disjoint update rects, each of which can cross a 512px boundary.
    const QVector<QRect> rects {QRect(-35, -17, 280, 535), QRect(245, -17, 310, 535)};
    for (const auto &r : rects) cpu.bltFixed(r, dabs);
    QVERIFY(KisGpuBrushPaint::tryPaint(&gpu, rects, dabs));
    const QByteArray wanted = readPixels(expected), got = readPixels(actual);
    int maxError = 0;
    for (int i = 0; i < wanted.size(); i += 4) {
        const int wa = quint8(wanted[i + 3]), ga = quint8(got[i + 3]);
        maxError = qMax(maxError, qAbs(wa - ga));
        if (alphaLocked) QCOMPARE(ga, wa);
        // Compare visible premultiplied colour. FMA/rounding differences in
        // straight RGB become large at tiny alpha; fully transparent RGB is
        // undefined in several SIMD selection paths and has no visible value.
        for (int c = 0; c < 3; ++c) {
            const int wc = (int(quint8(wanted[i + c])) * wa + 127) / 255;
            const int gc = (int(quint8(got[i + c])) * ga + 127) / 255;
            maxError = qMax(maxError, qAbs(wc - gc));
        }
    }
    QVERIFY2(maxError <= 2, qPrintable(QString("GPU differs from KisPainter by %1 displayed byte levels").arg(maxError)));
}

void KisGpuBrushPaintTest::testUndoRedo()
{
    auto device = makeDevice();
    const QByteArray before = readPixels(device);
    KisSurrogateUndoAdapter undo;
    KisTransaction transaction(kundo2_noi18n("GPU brush"), device, nullptr);
    KisPainter painter(device);
    QVERIFY(KisGpuBrushPaint::tryPaint(&painter, {testRect}, makeDabs()));
    const QByteArray after = readPixels(device);
    QVERIFY(after != before);
    transaction.commit(&undo);
    undo.undo();
    QCOMPARE(readPixels(device), before);
    undo.redo();
    QCOMPARE(readPixels(device), after);
}

void KisGpuBrushPaintTest::testUnsupportedMode()
{
    auto device = makeDevice();
    const QByteArray before = readPixels(device);
    KisPainter painter(device);
    painter.setCompositeOpId(COMPOSITE_MULT);
    const auto dabs = makeDabs();
    QVERIFY(!KisGpuBrushPaint::shouldUse(&painter, {testRect}, dabs));
    QVERIFY(!KisGpuBrushPaint::tryPaint(&painter, {testRect}, dabs));
    QCOMPARE(readPixels(device), before);
    KisPaintDeviceSP hdr = new KisPaintDevice(KoColorSpaceRegistry::instance()->rgb16());
    KisPainter hdrPainter(hdr);
    QVERIFY(!KisGpuBrushPaint::tryPaint(&hdrPainter, {testRect}, dabs));
}

void KisGpuBrushPaintTest::testLargeBatchTiming()
{
    const KoColorSpace *cs = KoColorSpaceRegistry::instance()->rgb8();
    KisFixedPaintDeviceSP tip = new KisFixedPaintDevice(cs);
    tip->setRect(QRect(0, 0, 512, 512));
    tip->initialize();
    for (int y = 0; y < 512; ++y) for (int x = 0; x < 512; ++x) {
        quint8 *p = tip->data() + (y * 512 + x) * 4;
        p[0] = 80; p[1] = 120; p[2] = 200;
        p[3] = quint8(qMax(0, 255 - int(std::hypot(x - 256, y - 256))));
    }
    QList<KisRenderedDab> dabs;
    for (int i = 0; i < 128; ++i) {
        KisRenderedDab dab(tip);
        dab.offset = QPoint(i % 7 - 3, i % 5 - 2);
        dab.opacity = 0.1;
        dabs.append(dab);
    }
    const QRect bounds(0, 0, 512, 512);
    KisPaintDeviceSP cpuDevice = new KisPaintDevice(cs), gpuDevice = new KisPaintDevice(cs);
    KisPainter cpu(cpuDevice), gpu(gpuDevice);
    QVERIFY(KisGpuBrushPaint::shouldUse(&gpu, {bounds}, dabs));
    // Warm the driver and the paint-device allocation before measuring.
    QVERIFY(KisGpuBrushPaint::tryPaint(&gpu, {bounds}, dabs));
    cpu.bltFixed(bounds, dabs);
    QElapsedTimer timer;
    timer.start();
    for (int i = 0; i < 5; ++i) cpu.bltFixed(bounds, dabs);
    const qint64 cpuTime = timer.nsecsElapsed();
    timer.restart();
    for (int i = 0; i < 5; ++i) QVERIFY(KisGpuBrushPaint::tryPaint(&gpu, {bounds}, dabs));
    const qint64 gpuTime = timer.nsecsElapsed();
    qInfo() << "128 overlapping 512px dabs, average batch ms: KisPainter"
            << cpuTime / 5000000.0 << "; GPU including packing/transfers/writeback"
            << gpuTime / 5000000.0;
}

SIMPLE_TEST_MAIN(KisGpuBrushPaintTest)
