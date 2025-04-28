"use client"

import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { loadSampleData } from "@/lib/utils"

interface SampleDataProviderProps {
  onSampleLoaded: (t1: HTMLImageElement, t2: HTMLImageElement, groundTruth: HTMLImageElement) => void
}

export default function SampleDataProvider({ onSampleLoaded }: SampleDataProviderProps) {
  const handleLoadSample = async () => {
    try {
      const { t1, t2, groundTruth } = await loadSampleData()
      onSampleLoaded(t1, t2, groundTruth)
    } catch (error) {
      console.error("Failed to load sample data:", error)
    }
  }

  return (
    <Card>
      <CardContent className="pt-6">
        <h3 className="text-lg font-medium mb-4">Sample Data</h3>
        <p className="text-sm text-gray-500 mb-4">
          Load sample data from the iSeg 2017 dataset to test the segmentation model.
        </p>
        <Button onClick={handleLoadSample} variant="outline" className="w-full">
          Load Sample Data
        </Button>
      </CardContent>
    </Card>
  )
}
