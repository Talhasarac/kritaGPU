/*
 * SPDX-FileCopyrightText: 2026 KritaGPU contributors
 * SPDX-License-Identifier: GPL-2.0-or-later
 */
#ifndef KIS_GPU_BRUSH_PAINT_H
#define KIS_GPU_BRUSH_PAINT_H

#include <QList>
#include <QRect>
#include <QVector>
#include <cstdint>

class KisPainter;
class KisPaintDevice;
struct KisRenderedDab;

namespace KisGpuBrushPaint
{
// Owned by one stroke/paintop and accessed by sequential painting jobs only.
// The device's mutation counter detects CPU fallback, undo and other writers.
class Session
{
public:
    Session();
    Session(const Session &) = delete;
    Session &operator=(const Session &) = delete;
    void prepare(KisPaintDevice *device);
    void committed(KisPaintDevice *device);
    void invalidate() { m_device = nullptr; }
    std::uint64_t id() const { return m_id; }
    std::uint64_t generation() const { return m_generation; }
private:
    const std::uint64_t m_id;
    std::uint64_t m_generation = 0;
    KisPaintDevice *m_device = nullptr;
    int m_sequence = -1;
    int m_lod = -1;
    int m_time = -1;
};
bool shouldUse(KisPainter *painter, const QVector<QRect> &rects, const QList<KisRenderedDab> &dabs);
// Bound the first asynchronous fetch as well as later adaptive updates, so a
// long stroke does not exceed the source-buffer budget before GPU admission.
int maxDabsPerUpdate(KisPainter *painter, int averageDiameter);
// Called by a sequential stroke job. False means the paint device is unchanged.
bool tryPaint(KisPainter *painter, const QVector<QRect> &rects, const QList<KisRenderedDab> &dabs,
              Session *session = nullptr);
}
#endif
