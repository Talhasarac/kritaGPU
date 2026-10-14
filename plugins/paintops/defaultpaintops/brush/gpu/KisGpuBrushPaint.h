/*
 * SPDX-FileCopyrightText: 2026 KritaGPU contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef KIS_GPU_BRUSH_PAINT_H
#define KIS_GPU_BRUSH_PAINT_H

#include <QList>
#include <QRect>
#include <QVector>

class KisPainter;
struct KisRenderedDab;

namespace KisGpuBrushPaint
{
bool shouldUse(KisPainter *painter, const QVector<QRect> &rects, const QList<KisRenderedDab> &dabs);
// Called by a sequential stroke job. False means the paint device is unchanged.
bool tryPaint(KisPainter *painter, const QVector<QRect> &rects, const QList<KisRenderedDab> &dabs);
}
#endif
