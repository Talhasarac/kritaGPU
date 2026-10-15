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
#include <kritaimage_export.h>
#include <memory>

class KisPainter;
class KisPaintDevice;
class KisFixedPaintDevice;
class KoColor;
class QString;
struct KisRenderedDab;

namespace KisGpuBrushPaint
{
// Owned by one stroke/paintop and accessed by sequential painting jobs only.
// The device's mutation counter detects CPU fallback, undo and other writers.
class KRITAIMAGE_EXPORT Session
{
public:
    Session();
    ~Session();
    Session(const Session &) = delete;
    Session &operator=(const Session &) = delete;
    void prepare(KisPaintDevice *device);
    void committed(KisPaintDevice *device);
    void invalidate() { m_device = nullptr; }
    std::uint64_t id() const { return m_id; }
    std::uint64_t generation() const { return m_generation; }
    void setPresentationKey(std::uint64_t key);
    std::uint64_t presentationKey() const;
    bool canDefer() const;
    bool hasPending() const;
    void enqueue(KisPainter *painter,const QVector<QRect> &rects,const QList<KisRenderedDab> &dabs);
    void flush();
    void discard();
private:
    struct Pending;
    std::unique_ptr<Pending> m_pending;
    const std::uint64_t m_id;
    std::uint64_t m_generation = 0;
    KisPaintDevice *m_device = nullptr;
    int m_sequence = -1;
    int m_lod = -1;
    int m_time = -1;
};
KRITAIMAGE_EXPORT bool shouldUse(KisPainter *painter, const QVector<QRect> &rects, const QList<KisRenderedDab> &dabs);
// Bound the first asynchronous fetch as well as later adaptive updates, so a
// long stroke does not exceed the source-buffer budget before GPU admission.
KRITAIMAGE_EXPORT int maxDabsPerUpdate(KisPainter *painter, int averageDiameter);
KRITAIMAGE_EXPORT bool hasDeferred(KisPaintDevice *device);
KRITAIMAGE_EXPORT void flushDevice(KisPaintDevice *device);
KRITAIMAGE_EXPORT void discardDevice(KisPaintDevice *device);
// Called by a sequential stroke job. False means the paint device is unchanged.
KRITAIMAGE_EXPORT bool tryPaint(KisPainter *painter, const QVector<QRect> &rects, const QList<KisRenderedDab> &dabs,
              Session *session = nullptr);
KRITAIMAGE_EXPORT bool trySmudge(KisPainter *painter, const QRect &rect,
    KisFixedPaintDevice *background, KisFixedPaintDevice *carried, KisFixedPaintDevice *mask,
    const KoColor &dulling, const KoColor &paint, bool useDulling, bool fused,
    const QString &smearOp, qreal smearOpacity, const QString &colourOp, qreal colourOpacity,
    Session *session);
}
#endif
