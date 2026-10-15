/*
 *  SPDX-FileCopyrightText: 2017 Dmitry Kazakov <dimula73@gmail.com>
 *
 *  SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "KisDabCacheUtils.h"

#include "kis_brush.h"
#include "kis_paint_device.h"
#include "kis_fixed_paint_device.h"
#include "kis_color_source.h"

#include <KisSharpnessOption.h>
#include <kis_texture_option.h>

#include <kundo2command.h>
#include <kis_auto_brush.h>
#include <kis_base_mask_generator.h>
#include <kis_image_config.h>
#include <KoColorModelStandardIds.h>
#include <QHash>
#include <QMutex>
#include <QMutexLocker>
#include <QTransform>
#include <QDebug>
#include <atomic>
#include <cmath>
#include <cstring>

namespace KisDabCacheUtils
{
namespace {
bool deferGpuDab(const DabGenerationInfo &di, DabRenderingResources *resources,
                 KisFixedPaintDeviceSP *device, bool imageStamp)
{
#ifndef Q_OS_WIN
    return false;
#else
    const KoColorSpace *cs = (*device)->colorSpace();
    static std::atomic<unsigned> logged {0};
    const bool logTip=qEnvironmentVariableIsSet("KRITA_GPU_BRUSH_LOG") && logged.fetch_add(1)<4;
    if (logTip) qInfo()<<"GPU tip candidate:"<<cs->id()<<di.solidColorFill<<KisImageConfig(true).useGpuBrushCompositing();
    if (cs->colorModelId() != RGBAColorModelID || !di.solidColorFill ||
        qEnvironmentVariableIsSet("KRITA_GPU_BRUSH_CPU_TIPS") ||
        !KisImageConfig(true).useGpuBrushCompositing() ||
        (cs->pixelSize() != 4 && cs->pixelSize() != 8 && cs->pixelSize() != 16)) return false;
    KisBrushSP brush = resources->brush;
    const int width = brush->maskWidth(di.shape,di.subPixel.x(),di.subPixel.y(),di.info);
    const int height = brush->maskHeight(di.shape,di.subPixel.x(),di.subPixel.y(),di.info);
    if (width < 32 || height < 32 || width > 8192 || height > 8192) return false;
    if (qint64(width)*height<65536 && !qEnvironmentVariableIsSet("KRITA_GPU_BRUSH_FORCE")) return false;
    auto tip = std::make_shared<KisDeferredGpuDab>();
    auto &d = tip->description;
    const QPointF centre = brush->hotSpot(di.shape,di.info) - QPointF(0.5,0.5) + di.subPixel;
    d.shape[0] = centre.x(); d.shape[1] = centre.y();
    const double angle = di.shape.rotation() + brush->angle();
    d.transform[0] = std::cos(angle); d.transform[1] = std::sin(angle);
    QVector<float> colour(4);
    const KoColor paint = di.paintColor.convertedTo(cs);
    cs->normalisedChannelsValue(paint.data(),colour);
    for (int c = 0; c < 4; ++c) d.colour[c] = colour[c] * 255.0f;
    if (auto *automatic = dynamic_cast<KisAutoBrush *>(brush.data())) {
        const auto *mask = automatic->maskGenerator();
        if (logTip) qInfo()<<"GPU automatic tip:"<<mask->id()<<int(mask->type())<<mask->spikes()
            <<automatic->randomness()<<automatic->density()<<width<<height;
        if ((mask->id() != DefaultId.id() && mask->id()!=GaussId.id()) || mask->type() != KisMaskGenerator::CIRCLE || mask->spikes() != 2 ||
            automatic->randomness() != 0 || automatic->density() != 1) return false;
        const double w = mask->diameter() * di.shape.scaleX();
        const double h = mask->diameter() * mask->ratio() * di.shape.scaleY();
        if (w < 10 || h < 10) return false;
        d.data[1] = 1; d.data[2] = mask->antialiasEdges() ? 1 : 0;
        d.shape[2] = 2.0 / w; d.shape[3] = 2.0 / h;
        const double softness = 1.0 / std::max(0.01,di.softnessFactor);
        d.transform[2] = (mask->horizontalFade() == 0 ? 1 : 2.0/(mask->horizontalFade()*w)) * softness;
        d.transform[3] = (mask->verticalFade() == 0 ? 1 : 2.0/(mask->verticalFade()*h)) * softness;
        if (mask->id()==GaussId.id()) {
            // Cache the original generator's one-dimensional radial profile.
            // The GPU expands it into rotated elliptical pixels; no full dab
            // bitmap is generated or uploaded, even for pressure-sized tips.
            const QString profileKey=QStringLiteral("%1:%2:%3:%4:%5")
                .arg(w,0,'g',17).arg(mask->horizontalFade(),0,'g',17)
                .arg(mask->verticalFade(),0,'g',17).arg(di.softnessFactor,0,'g',17)
                .arg(mask->antialiasEdges());
            static QMutex profileMutex;
            static QHash<QString,std::shared_ptr<const std::vector<std::uint32_t>>> profiles;
            QMutexLocker lock(&profileMutex);
            auto profile=profiles.value(profileKey);
            constexpr unsigned samples=4097;
            if (!profile) {
                std::unique_ptr<KisMaskGenerator> generator(mask->clone());
                generator->setSoftness(di.softnessFactor);
                generator->setScale(di.shape.scaleX(),di.shape.scaleY());
                auto values=std::make_shared<std::vector<std::uint32_t>>(samples);
                for (unsigned i=0;i<samples;++i)
                    (*values)[i]=255u-generator->valueAt(0.5*w*i/(samples-1),0);
                profile=values;
                if (profiles.size()>=64) profiles.clear();
                profiles.insert(profileKey,profile);
            }
            tip->source=profile;
            d.data[1]=6; d.data[2]=samples;
            d.shape[2]=2.0/w; d.shape[3]=w/h;
        }
    } else {
        if (brush->brushType() == PIPE_IMAGE || brush->brushType() == PIPE_MASK ||
            brush->preserveLightness() ||
            (brush->brushApplication() != ALPHAMASK && brush->brushApplication() != IMAGESTAMP && !imageStamp)) return false;
        const QImage originalImage = brush->brushTipImage();
        if (originalImage.isNull() || originalImage.sizeInBytes() > 16 * 1024 * 1024) return false;
        double mipScale=1.0;
        const double requestedScale=di.shape.scale()*brush->scale();
        while (mipScale*0.5>=requestedScale &&
               (originalImage.size()*(mipScale*0.5)).width()>0 &&
               (originalImage.size()*(mipScale*0.5)).height()>0) mipScale*=0.5;
        const QSize mipSize=originalImage.size()*mipScale;
        static QMutex cacheMutex;
        static QHash<QString,std::shared_ptr<const std::vector<std::uint32_t>>> images;
        QMutexLocker lock(&cacheMutex);
        const bool stamp = imageStamp || brush->brushApplication() == IMAGESTAMP;
        const auto cacheKey=QStringLiteral("%1:%2:%3:%4").arg(originalImage.cacheKey()).arg(mipSize.width())
            .arg(mipSize.height()).arg(stamp ? quint64(reinterpret_cast<quintptr>(cs)) : 0);
        auto pixels = images.value(cacheKey);
        if (!pixels) {
            QImage image=originalImage.convertToFormat(QImage::Format_ARGB32);
            if (image.size()!=mipSize) image=image.scaled(mipSize,Qt::IgnoreAspectRatio,Qt::SmoothTransformation);
            const unsigned words=stamp ? cs->pixelSize()/4 : 1;
            auto data = std::make_shared<std::vector<std::uint32_t>>(image.width()*image.height()*words);
            if (stamp) {
                KisFixedPaintDevice converted(cs);
                converted.convertFromQImage(image,QString());
                std::memcpy(data->data(),converted.constData(),data->size()*4);
            } else {
                for (int y=0;y<image.height();++y) std::memcpy(data->data()+y*image.width(),image.constScanLine(y),image.width()*4);
            }
            pixels = data;
            if (images.size() >= 16) images.clear();
            images.insert(cacheKey,pixels);
        }
        tip->source = pixels;
        d.data[1] = stamp ? 3 : 2;
        d.data[2] = mipSize.width();
        d.params[3] = mipSize.height();
        QTransform mapping;
        mapping.scale(di.shape.scaleX()*brush->scale(),di.shape.scaleY()*brush->scale());
        if (!qFuzzyIsNull(angle)) {
            mapping=mapping*QTransform().rotateRadians(-angle);
            const QRectF rotated=mapping.mapRect(QRectF(QPointF(),originalImage.size()));
            mapping=mapping*QTransform::fromTranslate(-rotated.x(),-rotated.y());
        }
        mapping=mapping*QTransform::fromTranslate(di.subPixel.x(),di.subPixel.y());
        bool invertible=false;
        const auto inverse=mapping.inverted(&invertible);
        if (!invertible) return false;
        const double mx=double(mipSize.width())/originalImage.width(), my=double(mipSize.height())/originalImage.height();
        d.shape[0]=inverse.m11()*mx; d.shape[1]=inverse.m21()*mx;
        d.shape[2]=inverse.m12()*my; d.shape[3]=inverse.m22()*my;
        d.transform[0]=inverse.dx()*mx-0.5; d.transform[1]=inverse.dy()*my-0.5;
        d.transform[2]=d.transform[3]=0;
    }
    if (!KisGpuBrush::Compositor::instance().isAvailable()) return false;
    const KisBrushSP snapshot = brush->clone().dynamicCast<KisBrush>();
    const DabGenerationInfo saved = di;
    tip->generateCpu = [snapshot,saved,cs,imageStamp](KisFixedPaintDevice *dst) {
        if (imageStamp || snapshot->brushApplication() == IMAGESTAMP) {
            *dst = *snapshot->paintDevice(cs,saved.shape,saved.info,saved.subPixel.x(),saved.subPixel.y());
        } else {
            KisFixedPaintDeviceSP generated = new KisFixedPaintDevice(cs);
            snapshot->mask(generated,saved.paintColor,saved.shape,saved.info,
                            saved.subPixel.x(),saved.subPixel.y(),saved.softnessFactor,saved.lightnessStrength);
            *dst = *generated;
        }
        if (!saved.mirrorProperties.isEmpty()) dst->mirror(saved.mirrorProperties.horizontalMirror,saved.mirrorProperties.verticalMirror);
    };
    d.data[3] = (di.mirrorProperties.horizontalMirror ? 1u : 0u) | (di.mirrorProperties.verticalMirror ? 2u : 0u);
    (*device)->setRect(QRect(0,0,width,height));
    (*device)->setDeferredGpuDab(std::move(tip));
    if (logTip) qInfo()<<"Deferred GPU tip:"<<d.data[1]<<width<<height;
    return true;
#endif
}
}

DabRenderingResources::DabRenderingResources()
{
}

DabRenderingResources::~DabRenderingResources()
{
}

void DabRenderingResources::syncResourcesToSeqNo(int seqNo, const KisPaintInformation &info)
{
    brush->prepareForSeqNo(info, seqNo);
}

QRect correctDabRectWhenFetchedFromCache(const QRect &dabRect,
                                         const QSize &realDabSize)
{
    int diffX = (realDabSize.width() - dabRect.width()) / 2;
    int diffY = (realDabSize.height() - dabRect.height()) / 2;

    return QRect(dabRect.x() - diffX, dabRect.y() - diffY,
                 realDabSize.width() , realDabSize.height());
}

void generateDab(const DabGenerationInfo &di, DabRenderingResources *resources, KisFixedPaintDeviceSP *dab, bool forceNormalizedRGBAImageStamp)
{
    KIS_SAFE_ASSERT_RECOVER_RETURN(*dab);
    if (deferGpuDab(di,resources,dab,forceNormalizedRGBAImageStamp)) return;
    const KoColorSpace *cs = (*dab)->colorSpace();


    if (forceNormalizedRGBAImageStamp || resources->brush->brushApplication() == IMAGESTAMP) {
        *dab = resources->brush->paintDevice(cs, di.shape, di.info,
                                             di.subPixel.x(),
                                             di.subPixel.y());

    } else if (di.solidColorFill) {
        resources->brush->mask(*dab,
                               di.paintColor,
                               di.shape,
                               di.info,
                               di.subPixel.x(), di.subPixel.y(),
                               di.softnessFactor,
                               di.lightnessStrength);
    }
    else {
        if (!resources->colorSourceDevice ||
            *cs != *resources->colorSourceDevice->colorSpace()) {

            resources->colorSourceDevice = new KisPaintDevice(cs);
        }
        else {
            resources->colorSourceDevice->clear();
        }

        QRect maskRect(QPoint(), di.dstDabRect.size());
        resources->colorSource->colorize(resources->colorSourceDevice, maskRect, di.info.pos().toPoint());
        resources->colorSourceDevice->convertTo(cs);

        resources->brush->mask(*dab, resources->colorSourceDevice,
                               di.shape,
                               di.info,
                               di.subPixel.x(), di.subPixel.y(),
                               di.softnessFactor,
                               di.lightnessStrength);
    }

    if (!di.mirrorProperties.isEmpty()) {
        (*dab)->mirror(di.mirrorProperties.horizontalMirror,
                       di.mirrorProperties.verticalMirror);
    }
}

void postProcessDab(KisFixedPaintDeviceSP dab,
                    const QPoint &dabTopLeft,
                    const KisPaintInformation& info,
                    DabRenderingResources *resources)
{
    if (resources->sharpnessOption) {
        resources->sharpnessOption->applyThreshold(dab, info);
    }

    if (resources->textureOption) {
        resources->textureOption->apply(dab, dabTopLeft, info);
    }
}

}

