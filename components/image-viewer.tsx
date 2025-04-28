"use client"

import { useEffect, useRef } from "react"
import type { Tensor3D } from "@tensorflow/tfjs"
import { tensorToCanvas } from "@/lib/utils"

interface ImageViewerProps {
  image?: HTMLImageElement | null
  tensor?: Tensor3D | null
  isSegmentation?: boolean
}

export default function ImageViewer({ image, tensor, isSegmentation = false }: ImageViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const ctx = canvas.getContext("2d")
    if (!ctx) return

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    if (image) {
      // Resize canvas to match image dimensions
      canvas.width = image.width
      canvas.height = image.height

      if (isSegmentation) {
        // For segmentation masks, use a colormap
        ctx.drawImage(image, 0, 0)
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height)
        const data = imageData.data

        // Apply colormap to segmentation
        for (let i = 0; i < data.length; i += 4) {
          const value = data[i] // Assuming grayscale image

          // Simple colormap:
          // 0 = background (transparent)
          // 1 = class 1 (red)
          // 2 = class 2 (green)
          // 3 = class 3 (blue)
          if (value === 0) {
            data[i + 3] = 0 // Transparent
          } else if (value === 1) {
            data[i] = 255 // Red
            data[i + 1] = 0
            data[i + 2] = 0
            data[i + 3] = 200 // Semi-transparent
          } else if (value === 2) {
            data[i] = 0
            data[i + 1] = 255 // Green
            data[i + 2] = 0
            data[i + 3] = 200
          } else if (value === 3) {
            data[i] = 0
            data[i + 1] = 0
            data[i + 2] = 255 // Blue
            data[i + 3] = 200
          }
        }

        ctx.putImageData(imageData, 0, 0)
      } else {
        // Regular image display
        ctx.drawImage(image, 0, 0)
      }
    } else if (tensor) {
      // Convert tensor to canvas
      tensorToCanvas(tensor, canvas, isSegmentation)
    }
  }, [image, tensor, isSegmentation])

  return (
    <div className="w-full aspect-square bg-gray-100 rounded-md overflow-hidden">
      <canvas ref={canvasRef} className="w-full h-full object-contain" />
    </div>
  )
}
