"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import type { AnalyzeVolume } from "@/lib/analyze-parser"

interface DebugInfoProps {
  t1Volume: AnalyzeVolume | null
  t2Volume: AnalyzeVolume | null
  groundTruthVolume: AnalyzeVolume | null
}

export default function DebugInfo({ t1Volume, t2Volume, groundTruthVolume }: DebugInfoProps) {
  const [isExpanded, setIsExpanded] = useState(false)

  if (!t1Volume && !t2Volume && !groundTruthVolume) {
    return null
  }

  return (
    <Card className="mt-4">
      <CardContent className="pt-6">
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-medium">Debug Information</h3>
          <Button variant="outline" size="sm" onClick={() => setIsExpanded(!isExpanded)}>
            {isExpanded ? "Hide Details" : "Show Details"}
          </Button>
        </div>

        {isExpanded && (
          <div className="space-y-4">
            {t1Volume && (
              <div>
                <h4 className="font-medium mb-1">T1 Volume</h4>
                <pre className="bg-gray-100 p-2 rounded-md text-xs overflow-x-auto">
                  {JSON.stringify(
                    {
                      dimensions: t1Volume.header.dimensions,
                      dataType: t1Volume.header.dataTypeString,
                      pixelDimensions: t1Volume.header.pixelDimensions,
                      min: t1Volume.min,
                      max: t1Volume.max,
                    },
                    null,
                    2,
                  )}
                </pre>
              </div>
            )}

            {t2Volume && (
              <div>
                <h4 className="font-medium mb-1">T2 Volume</h4>
                <pre className="bg-gray-100 p-2 rounded-md text-xs overflow-x-auto">
                  {JSON.stringify(
                    {
                      dimensions: t2Volume.header.dimensions,
                      dataType: t2Volume.header.dataTypeString,
                      pixelDimensions: t2Volume.header.pixelDimensions,
                      min: t2Volume.min,
                      max: t2Volume.max,
                    },
                    null,
                    2,
                  )}
                </pre>
              </div>
            )}

            {groundTruthVolume && (
              <div>
                <h4 className="font-medium mb-1">Ground Truth Volume</h4>
                <pre className="bg-gray-100 p-2 rounded-md text-xs overflow-x-auto">
                  {JSON.stringify(
                    {
                      dimensions: groundTruthVolume.header.dimensions,
                      dataType: groundTruthVolume.header.dataTypeString,
                      pixelDimensions: groundTruthVolume.header.pixelDimensions,
                      min: groundTruthVolume.min,
                      max: groundTruthVolume.max,
                    },
                    null,
                    2,
                  )}
                </pre>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  )
}
