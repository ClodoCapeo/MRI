"use client"

import type React from "react"

import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Upload } from "lucide-react"

interface ImageUploaderProps {
  onImageLoaded: (image: HTMLImageElement) => void
  imageType: string
  currentImage: HTMLImageElement | null
  optional?: boolean
}

export default function ImageUploader({
  onImageLoaded,
  imageType,
  currentImage,
  optional = false,
}: ImageUploaderProps) {
  const [isLoading, setIsLoading] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    setIsLoading(true)

    const reader = new FileReader()
    reader.onload = (event) => {
      const img = new Image()
      img.onload = () => {
        onImageLoaded(img)
        setIsLoading(false)
      }
      img.src = event.target?.result as string
    }
    reader.readAsDataURL(file)
  }

  const triggerFileInput = () => {
    fileInputRef.current?.click()
  }

  return (
    <div className="flex flex-col items-center">
      {currentImage ? (
        <div className="relative w-full aspect-square mb-4 bg-gray-100 rounded-md overflow-hidden">
          <img
            src={currentImage.src || "/placeholder.svg"}
            alt={`${imageType} image`}
            className="w-full h-full object-contain"
          />
        </div>
      ) : (
        <div
          className="w-full aspect-square mb-4 bg-gray-100 rounded-md flex items-center justify-center cursor-pointer hover:bg-gray-200 transition-colors"
          onClick={triggerFileInput}
        >
          <div className="flex flex-col items-center p-4">
            <Upload className="h-10 w-10 text-gray-400 mb-2" />
            <p className="text-sm text-gray-500 text-center">
              Click to upload {imageType} image
              {optional && " (optional)"}
            </p>
          </div>
        </div>
      )}

      <input type="file" ref={fileInputRef} onChange={handleFileChange} accept="image/*" className="hidden" />

      <Button
        onClick={triggerFileInput}
        variant={currentImage ? "outline" : "default"}
        size="sm"
        disabled={isLoading}
        className="w-full"
      >
        {isLoading ? "Loading..." : currentImage ? "Change Image" : "Upload Image"}
      </Button>
    </div>
  )
}
