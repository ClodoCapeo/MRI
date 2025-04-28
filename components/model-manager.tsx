"use client"

import type React from "react"

import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { InfoIcon } from "lucide-react"
import * as tf from "@tensorflow/tfjs"

interface ModelManagerProps {
  onModelLoaded: (modelPath: string) => void
  currentModelPath: string | null
}

export default function ModelManager({ onModelLoaded, currentModelPath }: ModelManagerProps) {
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileSelect = () => {
    fileInputRef.current?.click()
  }

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    setIsLoading(true)
    setError(null)
    setSuccess(null)

    try {
      // Create a unique model path for this upload
      const modelPath = `indexeddb://custom-unet-model-${Date.now()}`

      // For JSON format models
      if (file.name.endsWith(".json")) {
        // Read the model.json file
        const modelJSON = await file.text()
        const modelConfig = JSON.parse(modelJSON)

        // Check if this is a valid model file
        if (!modelConfig.modelTopology) {
          throw new Error("Invalid model file format")
        }

        // Look for weight files in the file list
        const weightFiles = Array.from(e.target.files || []).filter(
          (f) => f.name.endsWith(".bin") || f.name.includes("shard"),
        )

        if (weightFiles.length === 0) {
          throw new Error("No weight files found. Please select both model.json and weight files.")
        }

        // Create weight data
        const weightData = await Promise.all(
          weightFiles.map(async (wf) => {
            return { name: wf.name, data: new Uint8Array(await wf.arrayBuffer()) }
          }),
        )

        // Load the model
        const model = await tf.loadLayersModel(tf.io.fromMemory(modelConfig, weightData))

        // Save to IndexedDB for future use
        await model.save(modelPath)

        setSuccess(`Model loaded successfully with ${weightFiles.length} weight files`)
        onModelLoaded(modelPath)
      }
      // For binary format models
      else if (file.name.endsWith(".bin")) {
        throw new Error("Please select the model.json file instead of the .bin file")
      } else {
        throw new Error("Unsupported file format. Please upload a TensorFlow.js model (model.json)")
      }
    } catch (err) {
      console.error("Error loading model:", err)
      setError(`Error loading model: ${err instanceof Error ? err.message : String(err)}`)
    } finally {
      setIsLoading(false)
      if (fileInputRef.current) {
        fileInputRef.current.value = ""
      }
    }
  }

  return (
    <Card>
      <CardContent className="pt-6">
        <h3 className="text-xl font-bold mb-4">Custom Model Management</h3>
        <p className="text-sm text-gray-500 mb-4">
          Upload your own pre-trained U-Net model in TensorFlow.js format (model.json + weight files)
        </p>

        <div className="space-y-4">
          <div>
            <Button onClick={handleFileSelect} disabled={isLoading} className="w-full">
              {isLoading ? "Loading..." : "Upload Custom Model"}
            </Button>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              multiple
              accept=".json,.bin"
              className="hidden"
            />
            <p className="text-xs text-gray-500 mt-1">
              Select the model.json file and all associated weight files (.bin)
            </p>
          </div>

          {error && (
            <Alert variant="destructive">
              <InfoIcon className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {success && (
            <Alert variant="default" className="bg-green-50 border-green-200">
              <InfoIcon className="h-4 w-4 text-green-600" />
              <AlertDescription>{success}</AlertDescription>
            </Alert>
          )}

          {currentModelPath && (
            <div className="text-sm">
              <p className="font-medium">Current Model:</p>
              <p className="text-gray-500 truncate">
                {currentModelPath.startsWith("indexeddb://custom") ? "Custom Uploaded Model" : currentModelPath}
              </p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
