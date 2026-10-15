/*
 *  SPDX-FileCopyrightText: 2009 Boudewijn Rempt <boud@valdyas.org>
 *  SPDX-FileCopyrightText: 2010 Lukáš Tvrdý <lukast.dev@gmail.com>
 *
 *  SPDX-License-Identifier: GPL-2.0-or-later
 */
#include "kis_fixed_paint_device.h"

#include <KoColorSpaceRegistry.h>
#include <KoColor.h>
#include <KoColorModelStandardIds.h>
#include "kis_debug.h"

KisFixedPaintDevice::KisFixedPaintDevice(const KoColorSpace* colorSpace, KisOptimizedByteArray::MemoryAllocatorSP allocator)
        : m_colorSpace(colorSpace),
          m_data(allocator)
{
}

KisFixedPaintDevice::~KisFixedPaintDevice()
{
}

KisFixedPaintDevice::KisFixedPaintDevice(const KisFixedPaintDevice& rhs)
        : KisShared()
{
    std::lock_guard<std::recursive_mutex> lock(rhs.m_gpuMutex);
    m_bounds = rhs.m_bounds;
    m_colorSpace = rhs.m_colorSpace;
    m_data = rhs.m_data;
    m_gpuDab = rhs.m_gpuDab;
    m_gpuMaterialized = rhs.m_gpuMaterialized;
}

KisFixedPaintDevice& KisFixedPaintDevice::operator=(const KisFixedPaintDevice& rhs)
{
    if (this == &rhs) return *this;
    std::scoped_lock lock(m_gpuMutex,rhs.m_gpuMutex);
    m_bounds = rhs.m_bounds;
    m_colorSpace = rhs.m_colorSpace;


    const int referenceSize = m_bounds.height() * m_bounds.width() * pixelSize();

    m_gpuDab = rhs.m_gpuDab;
    m_gpuMaterialized = rhs.m_gpuMaterialized;
    if (m_gpuDab && !m_gpuMaterialized) {
        m_data.resize(0);
    } else if (m_data.size() >= referenceSize) {
        memcpy(m_data.data(), rhs.m_data.constData(), referenceSize);
    } else {
        m_data = rhs.m_data;
    }

    return *this;
}

void KisFixedPaintDevice::setRect(const QRect& rc)
{
    std::lock_guard<std::recursive_mutex> lock(m_gpuMutex);
    if (rc != m_bounds) m_gpuDab.reset();
    m_bounds = rc;
}

void KisFixedPaintDevice::setColorSpace(const KoColorSpace *cs)
{
    std::lock_guard<std::recursive_mutex> lock(m_gpuMutex);
    if (*m_colorSpace == *cs) { m_colorSpace=cs; return; }
    if (m_gpuDab && *m_colorSpace != *cs) materializeGpuDab();
    m_gpuDab.reset();
    m_colorSpace = cs;
}

QRect KisFixedPaintDevice::bounds() const
{
    std::lock_guard<std::recursive_mutex> lock(m_gpuMutex);
    return m_bounds;
}


int KisFixedPaintDevice::allocatedPixels() const
{
    std::lock_guard<std::recursive_mutex> lock(m_gpuMutex);
    if (m_gpuDab && !m_gpuMaterialized) return m_bounds.width() * m_bounds.height();
    return m_data.size() / m_colorSpace->pixelSize();
}


quint32 KisFixedPaintDevice::pixelSize() const
{
    return m_colorSpace->pixelSize();
}

bool KisFixedPaintDevice::initialize(quint8 defaultValue)
{
    m_gpuDab.reset();
    m_data.fill(defaultValue, m_bounds.height() * m_bounds.width() * pixelSize());

    return true;
}

void KisFixedPaintDevice::reallocateBufferWithoutInitialization()
{
    m_gpuDab.reset();
    const int referenceSize = m_bounds.height() * m_bounds.width() * pixelSize();

    if (referenceSize != m_data.size()) {
        m_data.resize(m_bounds.height() * m_bounds.width() * pixelSize());
    }
}

void KisFixedPaintDevice::lazyGrowBufferWithoutInitialization()
{
    m_gpuDab.reset();
    const int referenceSize = m_bounds.height() * m_bounds.width() * pixelSize();

    if (m_data.size() < referenceSize) {
        m_data.resize(referenceSize);
    }
}

quint8* KisFixedPaintDevice::data()
{
    std::lock_guard<std::recursive_mutex> lock(m_gpuMutex);
    materializeGpuDab();
    m_gpuDab.reset();
    return (quint8*) m_data.data();
}

const quint8 *KisFixedPaintDevice::constData() const
{
    std::lock_guard<std::recursive_mutex> lock(m_gpuMutex);
    materializeGpuDab();
    return (const quint8*) m_data.constData();
}

quint8* KisFixedPaintDevice::data() const
{
    std::lock_guard<std::recursive_mutex> lock(m_gpuMutex);
    materializeGpuDab();
    m_gpuDab.reset();
    return const_cast<quint8*>(m_data.constData());
}

void KisFixedPaintDevice::setDeferredGpuDab(std::shared_ptr<KisDeferredGpuDab> dab)
{
    std::lock_guard<std::recursive_mutex> lock(m_gpuMutex);
    m_gpuDab = std::move(dab);
    m_gpuMaterialized = false;
}

std::shared_ptr<const KisDeferredGpuDab> KisFixedPaintDevice::deferredGpuDab() const
{
    std::lock_guard<std::recursive_mutex> lock(m_gpuMutex);
    return m_gpuDab;
}

void KisFixedPaintDevice::materializeGpuDab() const
{
    std::lock_guard<std::recursive_mutex> lock(m_gpuMutex);
    if (!m_gpuDab || m_gpuMaterialized) return;
    auto dab = m_gpuDab;
    m_gpuDab.reset(); // permit the CPU generator to use data() recursively
    dab->generateCpu(const_cast<KisFixedPaintDevice *>(this));
    m_gpuDab = std::move(dab);
    m_gpuMaterialized = true;
}

void KisFixedPaintDevice::convertTo(const KoColorSpace* dstColorSpace,
                                    KoColorConversionTransformation::Intent renderingIntent,
                                    KoColorConversionTransformation::ConversionFlags conversionFlags)
{
    std::lock_guard<std::recursive_mutex> lock(m_gpuMutex);
    if (*m_colorSpace == *dstColorSpace) {
        return;
    }
    quint32 size = m_bounds.width() * m_bounds.height();
    KisOptimizedByteArray dstData(m_data.customMemoryAllocator());

    // make sure that we are not initializing the destination pixels!
    dstData.resize(size * dstColorSpace->pixelSize());

    m_colorSpace->convertPixelsTo(constData(), (quint8*)dstData.data(),
                                  dstColorSpace,
                                  size,
                                  renderingIntent,
                                  conversionFlags);

    m_colorSpace = dstColorSpace;
    m_data = dstData;
    m_gpuDab.reset();
}

void KisFixedPaintDevice::setProfile(const KoColorProfile *profile)
{
    std::lock_guard<std::recursive_mutex> lock(m_gpuMutex);
    materializeGpuDab();
    m_gpuDab.reset();
    KIS_SAFE_ASSERT_RECOVER_RETURN(profile);

    const KoColorSpace *dstColorSpace =
        KoColorSpaceRegistry::instance()->colorSpace(
                colorSpace()->colorModelId().id(),
                colorSpace()->colorDepthId().id(),
                profile);

    KIS_SAFE_ASSERT_RECOVER_RETURN(dstColorSpace);

    m_colorSpace = dstColorSpace;
}

void KisFixedPaintDevice::convertFromQImage(const QImage& _image, const QString &srcProfileName)
{
    QImage image = _image;

    if (image.format() != QImage::Format_ARGB32) {
        image.convertTo(QImage::Format_ARGB32);
    }
    setRect(image.rect());
    lazyGrowBufferWithoutInitialization();

    // Don't convert if not no profile is given and both paint dev and qimage are rgba.
    if (srcProfileName.isEmpty() && colorSpace()->id() == "RGBA") {
        memcpy(data(), image.constBits(), image.sizeInBytes());
    } else {
        KoColorSpaceRegistry::instance()
            ->colorSpace( RGBAColorModelID.id(), Integer8BitsColorDepthID.id(), srcProfileName)
            ->convertPixelsTo(image.constBits(), data(), colorSpace(), image.width() * image.height(),
                              KoColorConversionTransformation::internalRenderingIntent(),
                              KoColorConversionTransformation::internalConversionFlags());
    }
}

QImage KisFixedPaintDevice::convertToQImage(const KoColorProfile *  dstProfile, KoColorConversionTransformation::Intent intent, KoColorConversionTransformation::ConversionFlags conversionFlags) const
{
    qint32 x1;
    qint32 y1;
    qint32 w;
    qint32 h;

    x1 = m_bounds.x();
    y1 = m_bounds.y();
    w = m_bounds.width();
    h = m_bounds.height();

    return convertToQImage(dstProfile, x1, y1, w, h, intent, conversionFlags);
}

QImage KisFixedPaintDevice::convertToQImage(const KoColorProfile *  dstProfile, qint32 x1, qint32 y1, qint32 w, qint32 h, KoColorConversionTransformation::Intent intent, KoColorConversionTransformation::ConversionFlags conversionFlags) const
{
    Q_ASSERT( m_bounds.contains(QRect(x1,y1,w,h)) );

    if (w < 0)
        return QImage();

    if (h < 0)
        return QImage();

    if (QRect(x1, y1, w, h) == m_bounds) {
        return colorSpace()->convertToQImage(constData(), w, h, dstProfile,
                                             intent, conversionFlags);
    } else {
        try {
            // XXX: fill the image row by row!
            const int pSize = pixelSize();
            const int deviceWidth = m_bounds.width();
            quint8* newData = new quint8[w * h * pSize];
            const quint8* srcPtr = constData() + x1 * pSize + y1 * deviceWidth * pSize;
            quint8* dstPtr = newData;
            // copy the right area out of the paint device into data
            for (int row = 0; row < h; row++) {
                memcpy(dstPtr, srcPtr, w * pSize);
                srcPtr += deviceWidth * pSize;
                dstPtr += w * pSize;
            }
            QImage image = colorSpace()->convertToQImage(newData, w, h, dstProfile, intent, conversionFlags);
            return image;
        }
        catch(const std::bad_alloc&) {
            return QImage();
        }
    }
}

void KisFixedPaintDevice::clear(const QRect & rc)
{
    KoColor c(Qt::black, m_colorSpace);
    quint8* black = new quint8[pixelSize()];
    memcpy(black, c.data(), m_colorSpace->pixelSize());
    m_colorSpace->setOpacity(black, OPACITY_TRANSPARENT_U8, 1);
    fill(rc.x(), rc.y(), rc.width(), rc.height(), black);
    delete[] black;
}

void KisFixedPaintDevice::fill(const QRect &rc, const KoColor &color)
{
    KoColor realColor(color);
    realColor.convertTo(colorSpace());
    fill(rc.x(), rc.y(), rc.width(), rc.height(), realColor.data());
}

void KisFixedPaintDevice::fill(qint32 x, qint32 y, qint32 w, qint32 h, const quint8 *fillPixel)
{
    materializeGpuDab();
    m_gpuDab.reset();
    if (m_data.isEmpty() || m_bounds.isEmpty()) {
        setRect(QRect(x, y, w, h));
        reallocateBufferWithoutInitialization();
    }

    QRect rc(x, y, w, h);
    if (!m_bounds.contains(rc)) {
        rc = m_bounds;
    }

    quint8 pixelSize = m_colorSpace->pixelSize();
    quint8* dabPointer = data();

    if (rc.contains(m_bounds)) {
        for (int i = 0; i < w * h ; ++i) {
            memcpy(dabPointer, fillPixel, pixelSize);
            dabPointer += pixelSize;
        }

    } else {
        int deviceWidth = bounds().width();
        quint8* rowPointer = dabPointer + ((y - bounds().y()) * deviceWidth + (x - bounds().x())) * pixelSize;
        for (int row = 0; row < h; row++) {
            for (int col = 0; col < w; col++) {
                memcpy(rowPointer + col * pixelSize , fillPixel, pixelSize);
            }
            rowPointer += deviceWidth * pixelSize;
        }
    }
}


void KisFixedPaintDevice::readBytes(quint8* dstData, qint32 x, qint32 y, qint32 w, qint32 h) const
{
    materializeGpuDab();
    if (m_data.isEmpty() || m_bounds.isEmpty()) {
        return;
    }

    QRect rc(x, y, w, h);
    if (!m_bounds.contains(rc)){
        return;
    }

    const int pixelSize = m_colorSpace->pixelSize();
    const quint8* dabPointer = constData();

    if (rc == m_bounds) {
        memcpy(dstData, dabPointer, pixelSize * w * h);
    } else {
        int deviceWidth = m_bounds.width();
        const quint8* rowPointer = dabPointer + ((y - bounds().y()) * deviceWidth + (x - bounds().x())) * pixelSize;
        for (int row = 0; row < h; row++) {
            memcpy(dstData, rowPointer, w * pixelSize);
            rowPointer += deviceWidth * pixelSize;
            dstData += w * pixelSize;
        }
    }
}

void KisFixedPaintDevice::mirror(bool horizontal, bool vertical)
{
    if (auto original = deferredGpuDab()) {
        auto copy = std::make_shared<KisDeferredGpuDab>(*original);
        copy->description.data[3] ^= (horizontal ? 1u : 0u) | (vertical ? 2u : 0u);
        copy->generateCpu = [original,horizontal,vertical](KisFixedPaintDevice *dst) {
            original->generateCpu(dst);
            dst->mirror(horizontal,vertical);
        };
        setDeferredGpuDab(std::move(copy));
        return;
    }
    if (!horizontal && !vertical){
        return;
    }

    int pixelSize = m_colorSpace->pixelSize();
    int w = m_bounds.width();
    int h = m_bounds.height();

    if (horizontal){
        int rowSize = pixelSize * w;

        quint8 * dabPointer = data();
        quint8 * row = new quint8[ rowSize ];
        quint8 * mirror = 0;

        for (int y = 0; y < h ; y++){
            // TODO: implement better flipping of the data

            memcpy(row, dabPointer, rowSize);
            mirror = row;
            mirror += (w-1) * pixelSize;
            for (int x = 0; x < w; x++){
                memcpy(dabPointer,mirror,pixelSize);
                dabPointer += pixelSize;
                mirror -= pixelSize;
            }
        }

        delete [] row;
    }

    if (vertical){
        int rowsToMove = h / 2;
        int rowSize = pixelSize * w;

        quint8 * startRow = data();
        quint8 * endRow = data() + (h-1) * w * pixelSize;
        quint8 * row = new quint8[ rowSize ];

        for (int y = 0; y < rowsToMove; y++){
            memcpy(row, startRow, rowSize);
            memcpy(startRow, endRow, rowSize);
            memcpy(endRow, row, rowSize);

            startRow += rowSize;
            endRow -= rowSize;
        }

        delete [] row;
    }

}
