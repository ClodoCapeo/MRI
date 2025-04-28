"use client"

import { useState, useEffect, useRef } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Slider } from "@/components/ui/slider"
import * as tf from "@tensorflow/tfjs"
import type { AnalyzeVolume } from "@/lib/analyze-parser"

interface PostProcessingProps {
  segmentationVolume: AnalyzeVolume | null
  groundTruthVolume: AnalyzeVolume | null
  t1Volume: AnalyzeVolume | null
}

export default function PostProcessing({ segmentationVolume, groundTruthVolume, t1Volume }: PostProcessingProps) {
  const [processedVolume, setProcessedVolume] = useState<AnalyzeVolume | null>(null)
  const [sliceIndex, setSliceIndex] = useState(70)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const canvasOriginalRef = useRef<HTMLCanvasElement>(null)
  const canvasProcessedRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    if (!segmentationVolume) return

    // Apply argmax decoding as shown in the tutorial
    const applyArgmaxDecoding = async () => {
      try {
        // Convert volume data to tensor
        const segData = Array.from(segmentationVolume.data)
        const [width, height, depth, channels] = segmentationVolume.header.dimensions

        // Reshape to 4D tensor with channels as last dimension
        const segTensor = tf.tensor4d(segData, [width, height, depth, channels])

        // Apply argmax to get the class with highest probability
        const argmaxTensor = tf.argMax(segTensor, -1)

        // Convert back to volume
        const processedData = await argmaxTensor.data()

        setProcessedVolume({
          header: {
            ...segmentationVolume.header,
            dimensions: [width, height, depth],
          },
          data: new Float32Array(processedData),
          min: 0,
          max: 3,
        })

        // Clean up
        segTensor.dispose()
        argmaxTensor.dispose()
      } catch (error) {
        console.error("Error in post-processing:", error)
      }
    }

    applyArgmaxDecoding()
  }, [segmentationVolume])

  useEffect(() => {
    renderSlices()
  }, [sliceIndex, t1Volume, segmentationVolume, processedVolume, groundTruthVolume])

  const renderSlices = () => {
    if (!t1Volume) return

    // Render T1 slice
    const canvasT1 = canvasRef.current
    if (canvasT1) {
      const ctx = canvasT1.getContext("2d")
      if (ctx) {
        const { dimensions } = t1Volume.header
        const [width, height, depth] = dimensions

        canvasT1.width = width
        canvasT1.height = height

        // Extract slice data
        const sliceData = new Float32Array(width * height)
        for (let y = 0; y < height; y++) {
          for (let x = 0; x < width; x++) {
            const volumeIndex = x + y * width + sliceIndex * width * height
            sliceData[x + y * width] = t1Volume.data[volumeIndex]
          }
        }

        // Normalize and render
        const imageData = ctx.createImageData(width, height)
        let min = Number.POSITIVE_INFINITY
        let max = Number.NEGATIVE_INFINITY

        for (let i = 0; i < sliceData.length; i++) {
          min = Math.min(min, sliceData[i])
          max = Math.max(max, sliceData[i])
        }

        const range = max - min
        for (let i = 0; i < sliceData.length; i++) {
          const value = Math.round(((sliceData[i] - min) / range) * 255)
          imageData.data[i * 4] = value
          imageData.data[i * 4 + 1] = value
          imageData.data[i * 4 + 2] = value
          imageData.data[i * 4 + 3] = 255
        }

        ctx.putImageData(imageData, 0, 0)
      }
    }

    // Render original segmentation
    if (segmentationVolume) {
      const canvasOriginal = canvasOriginalRef.current
      if (canvasOriginal) {
        renderSegmentation(canvasOriginal, segmentationVolume, sliceIndex)
      }
    }

    // Render post-processed segmentation
    if (processedVolume) {
      const canvasProcessed = canvasProcessedRef.current
      if (canvasProcessed) {
        renderProcessedSegmentation(canvasProcessed, processedVolume, sliceIndex)
      }
    }
  }

  const renderSegmentation = (canvas: HTMLCanvasElement, volume: AnalyzeVolume, sliceIdx: number) => {
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const { dimensions } = volume.header
    const [width, height, depth] = dimensions

    canvas.width = width
    canvas.height = height

    // Create image data
    const imageData = ctx.createImageData(width, height)

    // Extract slice data and apply colormap
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const volumeIndex = x + y * width + sliceIdx * width * height
        const value = volume.data[volumeIndex]

        // Apply colormap based on class
        let r = 0,
          g = 0,
          b = 0,
          a = 0

        if (value === 0) {
          // Background - transparent
          r = 0
          g = 0
          b = 0
          a = 0
        } else if (value === 1) {
          // Class 1 - Blue (CSF)
          r = 65
          g = 105
          b = 225
          a = 200
        } else if (value === 2) {
          // Class 2 - Green (GM)
          r = 50
          g = 205
          b = 50
          a = 200
        } else if (value === 3 || value === 4) {
          // Class 3/4 - Orange (WM)
          r = 255
          g = 165
          b = 0
          a = 200
        }

        const pixelIndex = (y * width + x) * 4
        imageData.data[pixelIndex] = r
        imageData.data[pixelIndex + 1] = g
        imageData.data[pixelIndex + 2] = b
        imageData.data[pixelIndex + 3] = a
      }
    }

    ctx.putImageData(imageData, 0, 0)
  }

  const renderProcessedSegmentation = (canvas: HTMLCanvasElement, volume: AnalyzeVolume, sliceIdx: number) => {
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const { dimensions } = volume.header
    const [width, height, depth] = dimensions

    canvas.width = width
    canvas.height = height

    // Create image data
    const imageData = ctx.createImageData(width, height)

    // Extract slice data and apply colormap
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const volumeIndex = x + y * width + sliceIdx * width * height
        const value = volume.data[volumeIndex]

        // Apply colormap based on class
        let r = 0,
          g = 0,
          b = 0,
          a = 0

        if (value === 0) {
          // Background - transparent
          r = 0
          g = 0
          b = 0
          a = 0
        } else if (value === 1) {
          // Class 1 - Blue (CSF)
          r = 65
          g = 105
          b = 225
          a = 200
        } else if (value === 2) {
          // Class 2 - Green (GM)
          r = 50
          g = 205
          b = 50
          a = 200
        } else if (value === 3) {
          // Class 3 - Orange (WM)
          r = 255
          g = 165
          b = 0
          a = 200
        }

        const pixelIndex = (y * width + x) * 4
        imageData.data[pixelIndex] = r
        imageData.data[pixelIndex + 1] = g
        imageData.data[pixelIndex + 2] = b
        imageData.data[pixelIndex + 3] = a
      }
    }

    ctx.putImageData(imageData, 0, 0)
  }

  if (!t1Volume) return null

  return (
    <Card>
      <CardContent className="pt-6">
        <h3 className="text-lg font-medium mb-4">Post-Processing Results</h3>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Original T1 */}
          <div>
            <h4 className="text-sm font-medium mb-2">T1 Image</h4>
            <div className="bg-gray-100 rounded-md aspect-square relative">
              <canvas ref={canvasRef} className="w-full h-full" />
            </div>
          </div>

          {/* Raw Segmentation */}
          <div>
            <h4 className="text-sm font-medium mb-2">Raw Segmentation</h4>
            <div className="bg-gray-100 rounded-md aspect-square relative">
              <canvas ref={canvasOriginalRef} className="w-full h-full" />
            </div>
          </div>

          {/* Post-processed Segmentation */}
          <div>
            <h4 className="text-sm font-medium mb-2">Post-processed</h4>
            <div className="bg-gray-100 rounded-md aspect-square relative">
              <canvas ref={canvasProcessedRef} className="w-full h-full" />
            </div>
          </div>
        </div>

        <div className="mt-4">
          <label className="block text-sm font-medium mb-1">Slice: {sliceIndex}</label>
          <Slider
            value={[sliceIndex]}
            min={0}
            max={t1Volume.header.dimensions[2] - 1}
            step={1}
            onValueChange={(value) => setSliceIndex(value[0])}
          />
        </div>
      </CardContent>
    </Card>
  )
}
