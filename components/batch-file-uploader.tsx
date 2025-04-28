"use client"

import type React from "react"

import { useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { parseAnalyzeVolume } from "@/lib/analyze-parser"
import type { AnalyzeVolume } from "@/lib/analyze-parser"

interface SubjectFiles {
  t1Hdr?: File
  t1Img?: File
  t2Hdr?: File
  t2Img?: File
  labelHdr?: File
  labelImg?: File
}

interface Subject {
  id: string
  t1Volume?: AnalyzeVolume
  t2Volume?: AnalyzeVolume
  labelVolume?: AnalyzeVolume
  files: SubjectFiles
  isComplete: boolean
  isLoaded: boolean
}

interface BatchFileUploaderProps {
  onSubjectsLoaded: (subjects: Subject[], forTraining: boolean) => void
}

export default function BatchFileUploader({ onSubjectsLoaded }: BatchFileUploaderProps) {
  const [subjects, setSubjects] = useState<{ [key: string]: Subject }>({})
  const [isLoading, setIsLoading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileSelect = () => {
    fileInputRef.current?.click()
  }

  const handleFilesChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files || files.length === 0) return

    setIsLoading(true)
    setError(null)
    setProgress(0)

    try {
      // Group files by subject
      const newSubjects = { ...subjects }
      const fileArray = Array.from(files)

      // Parse file names to identify subjects and modalities
      for (const file of fileArray) {
        // Expected format: subject-[number]-[T1/T2/label].[hdr/img]
        const match = file.name.match(/subject-(\d+)-([^.]+)\.([^.]+)$/)
        if (!match) {
          console.warn(`File ${file.name} doesn't match the expected naming pattern, skipping`)
          continue
        }

        const [_, subjectId, modality, extension] = match
        const subjectKey = `subject-${subjectId}`

        // Initialize subject if it doesn't exist
        if (!newSubjects[subjectKey]) {
          newSubjects[subjectKey] = {
            id: subjectKey,
            files: {},
            isComplete: false,
            isLoaded: false,
          }
        }

        // Add file to the appropriate category
        if (modality.toLowerCase() === "t1" && extension.toLowerCase() === "hdr") {
          newSubjects[subjectKey].files.t1Hdr = file
        } else if (modality.toLowerCase() === "t1" && extension.toLowerCase() === "img") {
          newSubjects[subjectKey].files.t1Img = file
        } else if (modality.toLowerCase() === "t2" && extension.toLowerCase() === "hdr") {
          newSubjects[subjectKey].files.t2Hdr = file
        } else if (modality.toLowerCase() === "t2" && extension.toLowerCase() === "img") {
          newSubjects[subjectKey].files.t2Img = file
        } else if (modality.toLowerCase() === "label" && extension.toLowerCase() === "hdr") {
          newSubjects[subjectKey].files.labelHdr = file
        } else if (modality.toLowerCase() === "label" && extension.toLowerCase() === "img") {
          newSubjects[subjectKey].files.labelImg = file
        }

        // Check if subject has complete T1 and T2 files
        newSubjects[subjectKey].isComplete =
          !!newSubjects[subjectKey].files.t1Hdr &&
          !!newSubjects[subjectKey].files.t1Img &&
          !!newSubjects[subjectKey].files.t2Hdr &&
          !!newSubjects[subjectKey].files.t2Img
      }

      setSubjects(newSubjects)
    } catch (err) {
      setError(`Error processing files: ${err instanceof Error ? err.message : String(err)}`)
    } finally {
      setIsLoading(false)
      setProgress(100)
      // Reset the file input
      if (fileInputRef.current) {
        fileInputRef.current.value = ""
      }
    }
  }

  const loadSubjects = async (forTraining: boolean) => {
    setIsLoading(true)
    setError(null)
    setProgress(0)

    try {
      const subjectArray = Object.values(subjects).filter((subject) => subject.isComplete && !subject.isLoaded)
      const totalSubjects = subjectArray.length

      if (totalSubjects === 0) {
        setError("No complete subjects to load")
        setIsLoading(false)
        return
      }

      const loadedSubjects: Subject[] = []

      for (let i = 0; i < totalSubjects; i++) {
        const subject = subjectArray[i]
        setProgress(((i + 0.1) / totalSubjects) * 100)

        try {
          // Load T1 volume
          const t1HeaderBuffer = await subject.files.t1Hdr!.arrayBuffer()
          const t1ImageBuffer = await subject.files.t1Img!.arrayBuffer()
          const t1Volume = await parseAnalyzeVolume(t1HeaderBuffer, t1ImageBuffer)
          setProgress(((i + 0.4) / totalSubjects) * 100)

          // Load T2 volume
          const t2HeaderBuffer = await subject.files.t2Hdr!.arrayBuffer()
          const t2ImageBuffer = await subject.files.t2Img!.arrayBuffer()
          const t2Volume = await parseAnalyzeVolume(t2HeaderBuffer, t2ImageBuffer)
          setProgress(((i + 0.7) / totalSubjects) * 100)

          // Load label volume if available
          let labelVolume: AnalyzeVolume | undefined
          if (subject.files.labelHdr && subject.files.labelImg) {
            const labelHeaderBuffer = await subject.files.labelHdr.arrayBuffer()
            const labelImageBuffer = await subject.files.labelImg.arrayBuffer()
            labelVolume = await parseAnalyzeVolume(labelHeaderBuffer, labelImageBuffer)
          }
          setProgress(((i + 1) / totalSubjects) * 100)

          // Update subject with loaded volumes
          const loadedSubject: Subject = {
            ...subject,
            t1Volume,
            t2Volume,
            labelVolume,
            isLoaded: true,
          }

          loadedSubjects.push(loadedSubject)

          // Update subjects state
          setSubjects((prev) => ({
            ...prev,
            [subject.id]: loadedSubject,
          }))
        } catch (err) {
          console.error(`Error loading subject ${subject.id}:`, err)
          setError(`Error loading subject ${subject.id}: ${err instanceof Error ? err.message : String(err)}`)
        }
      }

      // Call the callback with loaded subjects
      if (loadedSubjects.length > 0) {
        onSubjectsLoaded(loadedSubjects, forTraining)
      }
    } catch (err) {
      setError(`Error loading subjects: ${err instanceof Error ? err.message : String(err)}`)
    } finally {
      setIsLoading(false)
      setProgress(100)
    }
  }

  const clearSubjects = () => {
    setSubjects({})
  }

  const completeSubjects = Object.values(subjects).filter((subject) => subject.isComplete)
  const loadedSubjects = Object.values(subjects).filter((subject) => subject.isLoaded)
  const remainingSubjects = completeSubjects.length - loadedSubjects.length

  return (
    <Card>
      <CardContent className="pt-6">
        <h3 className="text-xl font-bold mb-4">Batch Subject Loader</h3>
        <p className="text-sm text-gray-500 mb-4">
          Select multiple .hdr and .img files for T1, T2, and labels. Files should follow the naming pattern:
          subject-[number]-[T1/T2/label].[hdr/img]
        </p>

        <div className="space-y-4">
          <div>
            <Button onClick={handleFileSelect} disabled={isLoading} className="w-full">
              Select Files
            </Button>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFilesChange}
              multiple
              accept=".hdr,.img"
              className="hidden"
            />
          </div>

          {Object.keys(subjects).length > 0 && (
            <div className="mt-4">
              <h4 className="font-medium mb-2">Subjects ({Object.keys(subjects).length})</h4>
              <div className="bg-gray-100 p-3 rounded-md max-h-60 overflow-y-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr>
                      <th className="text-left">Subject</th>
                      <th className="text-center">T1</th>
                      <th className="text-center">T2</th>
                      <th className="text-center">Label</th>
                      <th className="text-center">Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.values(subjects).map((subject) => (
                      <tr key={subject.id} className="border-t border-gray-200">
                        <td className="py-2">{subject.id}</td>
                        <td className="text-center">{subject.files.t1Hdr && subject.files.t1Img ? "✅" : "❌"}</td>
                        <td className="text-center">{subject.files.t2Hdr && subject.files.t2Img ? "✅" : "❌"}</td>
                        <td className="text-center">
                          {subject.files.labelHdr && subject.files.labelImg ? "✅" : "❌"}
                        </td>
                        <td className="text-center">
                          {subject.isLoaded ? "Loaded" : subject.isComplete ? "Ready" : "Incomplete"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {isLoading && (
            <div className="mt-4">
              <Progress value={progress} className="h-2" />
              <p className="text-xs text-gray-500 mt-1">Processing: {Math.round(progress)}%</p>
            </div>
          )}

          {error && <div className="text-red-500 text-sm mt-2">{error}</div>}

          <div className="flex flex-col space-y-2 mt-4">
            <div className="flex justify-between text-sm mb-1">
              <span>Complete subjects: {completeSubjects.length}</span>
              <span>Loaded: {loadedSubjects.length}</span>
            </div>

            <div className="grid grid-cols-2 gap-2">
              <Button
                onClick={() => loadSubjects(true)}
                disabled={isLoading || remainingSubjects === 0}
                variant="default"
              >
                Load for Training
              </Button>
              <Button
                onClick={() => loadSubjects(false)}
                disabled={isLoading || remainingSubjects === 0}
                variant="outline"
              >
                Load for Validation
              </Button>
            </div>

            <Button
              onClick={clearSubjects}
              disabled={isLoading || Object.keys(subjects).length === 0}
              variant="outline"
            >
              Clear All
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
