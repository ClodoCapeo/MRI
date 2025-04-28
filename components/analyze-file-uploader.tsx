"use client"

import type React from "react"

import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { parseAnalyzeVolume } from "@/lib/analyze-parser"
import type { AnalyzeVolume } from "@/lib/analyze-parser"

interface AnalyzeFileUploaderProps {
  onVolumeLoaded: (volume: AnalyzeVolume) => void
  volumeType: string
  currentVolume: AnalyzeVolume | null
  optional?: boolean
}

export default function AnalyzeFileUploader({
  onVolumeLoaded,
  volumeType,
  currentVolume,
  optional = false,
}: AnalyzeFileUploaderProps) {
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const headerInputRef = useRef<HTMLInputElement>(null)
  const imageInputRef = useRef<HTMLInputElement>(null)
  const [headerFile, setHeaderFile] = useState<File | null>(null)
  const [imageFile, setImageFile] = useState<File | null>(null)

  const handleHeaderFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    if (!file.name.endsWith(".hdr")) {
      setError("Please select a valid .hdr file")
      return
    }

    setHeaderFile(file)
    setError(null)

    // Try to automatically find the matching .img file
    const imgFileName = file.name.replace(".hdr", ".img")
    if (imageFile?.name !== imgFileName) {
      // Clear the current image file if it doesn't match
      setImageFile(null)
    }
  }

  const handleImageFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    if (!file.name.endsWith(".img")) {
      setError("Please select a valid .img file")
      return
    }

    setImageFile(file)
    setError(null)

    // Try to automatically find the matching .hdr file
    const hdrFileName = file.name.replace(".img", ".hdr")
    if (headerFile?.name !== hdrFileName) {
      // Clear the current header file if it doesn't match
      setHeaderFile(null)
    }
  }

  const loadVolume = async () => {
    if (!headerFile || !imageFile) {
      setError("Please select both .hdr and .img files")
      return
    }

    setIsLoading(true)
    setError(null)

    try {
      const headerBuffer = await headerFile.arrayBuffer()
      const imageBuffer = await imageFile.arrayBuffer()

      const volume = await parseAnalyzeVolume(headerBuffer, imageBuffer)
      onVolumeLoaded(volume)
    } catch (err) {
      console.error("Error loading volume:", err)
      setError(`Error loading volume: ${err instanceof Error ? err.message : String(err)}`)
    } finally {
      setIsLoading(false)
    }
  }

  const triggerHeaderFileInput = () => {
    headerInputRef.current?.click()
  }

  const triggerImageFileInput = () => {
    imageInputRef.current?.click()
  }

  return (
    <div className="flex flex-col items-center">
      <div className="w-full mb-4 bg-gray-100 rounded-md overflow-hidden">
        <div className="p-4">
          <div className="flex flex-col gap-4">
            <div>
              <p className="text-sm font-medium mb-2">Header File (.hdr)</p>
              <div className="flex items-center gap-2">
                <Button
                  onClick={triggerHeaderFileInput}
                  variant="outline"
                  size="sm"
                  className="w-full"
                  disabled={isLoading}
                >
                  {headerFile ? headerFile.name : "Select .hdr file"}
                </Button>
                <input
                  type="file"
                  ref={headerInputRef}
                  onChange={handleHeaderFileChange}
                  accept=".hdr"
                  className="hidden"
                />
              </div>
            </div>

            <div>
              <p className="text-sm font-medium mb-2">Image File (.img)</p>
              <div className="flex items-center gap-2">
                <Button
                  onClick={triggerImageFileInput}
                  variant="outline"
                  size="sm"
                  className="w-full"
                  disabled={isLoading}
                >
                  {imageFile ? imageFile.name : "Select .img file"}
                </Button>
                <input
                  type="file"
                  ref={imageInputRef}
                  onChange={handleImageFileChange}
                  accept=".img"
                  className="hidden"
                />
              </div>
            </div>
          </div>

          {error && <div className="mt-4 text-sm text-red-500">{error}</div>}

          <Button
            onClick={loadVolume}
            disabled={isLoading || !headerFile || !imageFile}
            className="w-full mt-4"
            size="sm"
          >
            {isLoading ? "Loading..." : currentVolume ? "Replace Volume" : "Load Volume"}
          </Button>
        </div>
      </div>

      {currentVolume && (
        <div className="text-sm text-gray-500">
          Loaded: {volumeType} ({currentVolume.header.dimensions.join(" × ")})
        </div>
      )}
    </div>
  )
}
