"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import type { AnalyzeVolume } from "@/lib/analyze-parser"

interface DatasetItem {
  t1: AnalyzeVolume
  t2: AnalyzeVolume
  groundTruth?: AnalyzeVolume
}

interface DatasetManagerProps {
  trainingData: DatasetItem[]
  validationData: DatasetItem[]
  onRemoveTrainingItem: (index: number) => void
  onRemoveValidationItem: (index: number) => void
  onClearTrainingData: () => void
  onClearValidationData: () => void
  onLoadItem: (item: DatasetItem) => void
}

export default function DatasetManager({
  trainingData,
  validationData,
  onRemoveTrainingItem,
  onRemoveValidationItem,
  onClearTrainingData,
  onClearValidationData,
  onLoadItem,
}: DatasetManagerProps) {
  const [activeTab, setActiveTab] = useState("training")

  const renderDatasetTable = (data: DatasetItem[], onRemove: (index: number) => void) => {
    if (data.length === 0) {
      return <p className="text-gray-500 text-center py-4">No data available</p>
    }

    return (
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b">
              <th className="text-left py-2">#</th>
              <th className="text-left py-2">T1 Dimensions</th>
              <th className="text-left py-2">T2 Dimensions</th>
              <th className="text-left py-2">Ground Truth</th>
              <th className="text-right py-2">Actions</th>
            </tr>
          </thead>
          <tbody>
            {data.map((item, index) => (
              <tr key={index} className="border-b">
                <td className="py-2">{index + 1}</td>
                <td className="py-2">{item.t1.header.dimensions.join(" × ")}</td>
                <td className="py-2">{item.t2.header.dimensions.join(" × ")}</td>
                <td className="py-2">{item.groundTruth ? "✅" : "❌"}</td>
                <td className="py-2 text-right">
                  <Button variant="ghost" size="sm" onClick={() => onLoadItem(item)}>
                    Load
                  </Button>
                  <Button variant="ghost" size="sm" onClick={() => onRemove(index)}>
                    Remove
                  </Button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    )
  }

  return (
    <Card>
      <CardContent className="pt-6">
        <h3 className="text-xl font-bold mb-4">Dataset Manager</h3>

        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="training">Training Data ({trainingData.length})</TabsTrigger>
            <TabsTrigger value="validation">Validation Data ({validationData.length})</TabsTrigger>
          </TabsList>

          <TabsContent value="training" className="space-y-4">
            {renderDatasetTable(trainingData, onRemoveTrainingItem)}
            {trainingData.length > 0 && (
              <Button variant="outline" size="sm" onClick={onClearTrainingData} className="mt-2">
                Clear All Training Data
              </Button>
            )}
          </TabsContent>

          <TabsContent value="validation" className="space-y-4">
            {renderDatasetTable(validationData, onRemoveValidationItem)}
            {validationData.length > 0 && (
              <Button variant="outline" size="sm" onClick={onClearValidationData} className="mt-2">
                Clear All Validation Data
              </Button>
            )}
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  )
}
