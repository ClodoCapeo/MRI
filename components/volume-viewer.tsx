"use client"

import { useState, useEffect, useRef } from "react"
import { Slider } from "@/components/ui/slider"
import { Button } from "@/components/ui/button"
import { extractSlice, normalizeSlice, sliceToImageData, segmentationColormap } from "@/lib/analyze-parser"
import type { AnalyzeVolume } from "@/lib/analyze-parser"

interface VolumeViewerProps {
  volume: AnalyzeVolume
  isSegmentation?: boolean
  overlayVolume?: AnalyzeVolume
}

export default function VolumeViewer({ volume, isSegmentation = false, overlayVolume }: VolumeViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [sliceIndex, setSliceIndex] = useState(0)
  const [axis, setAxis] = useState<0 | 1 | 2>(2) // 0: Sagittal, 1: Coronal, 2: Axial
  const [maxSliceIndex, setMaxSliceIndex] = useState(0)
  // Add a state to track which segmentation layer to show
  const [segLayer, setSegLayer] = useState<number>(3) // 1: background, 2: white matter+bg, 3: all

  // Add a function to render the segmentation with layers
  // Add this function before the useEffect that renders the slices
  const getSegmentationColormap = (value: number, layer: number) => {
    // For layer 1, only show background (transparent)
    if (layer === 1) {
      return value === 0 ? [0, 0, 0, 0] : [0, 0, 0, 0]
    }

    // For layer 2, show background and white matter (class 3)
    if (layer === 2) {
      if (value === 0) return [0, 0, 0, 0] // Background (transparent)
      if (value === 3) return [255, 165, 0, 200] // WM (orange, semi-transparent)
      return [0, 0, 0, 0] // Hide other classes
    }

    // For layer 3 (default), show all classes
    switch (value) {
      case 0:
        return [0, 0, 0, 0] // Background (transparent)
      case 1:
        return [65, 105, 225, 200] // CSF (royal blue, semi-transparent)
      case 2:
        return [50, 205, 50, 200] // GM (lime green, semi-transparent)
      case 3:
        return [255, 165, 0, 200] // WM (orange, semi-transparent)
      default:
        return [255, 0, 0, 200] // Error (red, semi-transparent)
    }
  }

  useEffect(() => {
    // Update max slice index when volume or axis changes
    const dimensions = volume.header.dimensions
    setMaxSliceIndex(dimensions[axis] - 1)

    // Reset slice index if it's out of bounds
    if (sliceIndex >= dimensions[axis]) {
      setSliceIndex(Math.floor(dimensions[axis] / 2))
    }
  }, [volume, axis, sliceIndex])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Extract slice
    const slice = extractSlice(volume, sliceIndex, axis)

    // Set canvas dimensions
    canvas.width = slice.width
    canvas.height = slice.height

    // Normalize slice data
    const normalizedSlice = normalizeSlice(slice.data, volume.min, volume.max)

    // Convert to image data
    const imageData = sliceToImageData(
      { ...slice, data: normalizedSlice },
      isSegmentation ? segmentationColormap : undefined,
    )

    // Draw base image
    ctx.putImageData(imageData, 0, 0)

    // Draw overlay if available
    if (overlayVolume) {
      const overlaySlice = extractSlice(overlayVolume, sliceIndex, axis)

      // Only proceed if dimensions match
      if (overlaySlice.width === slice.width && overlaySlice.height === slice.height) {
        const normalizedOverlay = normalizeSlice(overlaySlice.data, overlayVolume.min, overlayVolume.max)

        // Create custom colormap based on selected layer
        const layerColormap = (value: number) => {
          const scaledValue = Math.round(value * 3) // Scale to 0-3 range
          return getSegmentationColormap(scaledValue, segLayer)
        }

        const overlayImageData = sliceToImageData({ ...overlaySlice, data: normalizedOverlay }, layerColormap)

        // Create a temporary canvas for the overlay
        const tempCanvas = document.createElement("canvas")
        tempCanvas.width = slice.width
        tempCanvas.height = slice.height
        const tempCtx = tempCanvas.getContext("2d")

        if (tempCtx) {
          tempCtx.putImageData(overlayImageData, 0, 0)

          // Draw the overlay with transparency
          ctx.globalAlpha = 0.7
          ctx.drawImage(tempCanvas, 0, 0)
          ctx.globalAlpha = 1.0
        }
      }
    }
  }, [volume, overlayVolume, sliceIndex, axis, isSegmentation, segLayer])

  const handleSliceChange = (value: number[]) => {
    setSliceIndex(value[0])
  }

  const handleAxisChange = (newAxis: 0 | 1 | 2) => {
    setAxis(newAxis)
    // Set slice to middle of the new axis
    setSliceIndex(Math.floor(volume.header.dimensions[newAxis] / 2))
  }

  return (
    <div className="flex flex-col items-center">
      <div className="w-full aspect-square bg-gray-100 rounded-md overflow-hidden">
        <canvas ref={canvasRef} className="w-full h-full object-contain" />
      </div>

      <div className="w-full mt-4">
        <div className="flex justify-between text-sm mb-1">
          <span>
            Slice: {sliceIndex + 1}/{maxSliceIndex + 1}
          </span>
        </div>
        <Slider value={[sliceIndex]} min={0} max={maxSliceIndex} step={1} onValueChange={handleSliceChange} />
      </div>

      <div className="flex gap-2 mt-4">
        <Button variant={axis === 0 ? "default" : "outline"} size="sm" onClick={() => handleAxisChange(0)}>
          Sagittal
        </Button>
        <Button variant={axis === 1 ? "default" : "outline"} size="sm" onClick={() => handleAxisChange(1)}>
          Coronal
        </Button>
        <Button variant={axis === 2 ? "default" : "outline"} size="sm" onClick={() => handleAxisChange(2)}>
          Axial
        </Button>
      </div>

      {/* Add layer selection buttons below the axis buttons */}
      <div className="flex gap-2 mt-4">
        <Button variant={segLayer === 1 ? "default" : "outline"} size="sm" onClick={() => setSegLayer(1)}>
          Background
        </Button>
        <Button variant={segLayer === 2 ? "default" : "outline"} size="sm" onClick={() => setSegLayer(2)}>
          WM + BG
        </Button>
        <Button variant={segLayer === 3 ? "default" : "outline"} size="sm" onClick={() => setSegLayer(3)}>
          All Tissues
        </Button>
      </div>
    </div>
  )
}
