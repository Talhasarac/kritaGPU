/*
 * SPDX-FileCopyrightText: 2026 KritaGPU contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef KIS_GPU_BRUSH_PAINT_TEST_H
#define KIS_GPU_BRUSH_PAINT_TEST_H
#include <QObject>
class KisGpuBrushPaintTest : public QObject
{
    Q_OBJECT
private Q_SLOTS:
    void initTestCase();
    void testAgainstPainter_data();
    void testAgainstPainter();
    void testUndoRedo();
    void testUnsupportedMode();
    void testLargeBatchTiming();
};
#endif
