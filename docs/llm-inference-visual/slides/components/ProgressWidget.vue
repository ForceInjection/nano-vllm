<script setup lang="ts">
import { ref, onMounted, watch } from 'vue'

const totalPages = ref(0)
const visitedPages = ref(new Set<number>())
const completedTests = ref(new Set<string>())
const expanded = ref(false)

const storageKey = 'nanovllm-progress'

onMounted(() => {
  // Load saved progress
  const saved = localStorage.getItem(storageKey)
  if (saved) {
    const data = JSON.parse(saved)
    visitedPages.value = new Set(data.visitedPages || [])
    completedTests.value = new Set(data.completedTests || [])
  }

  // Count total slides
  const slideCount = document.querySelectorAll('.slidev-page').length
  totalPages.value = slideCount || 16
})

// Listen for slide changes
if (typeof window !== 'undefined') {
  window.addEventListener('slidev:navigation', () => {
    // Track current page
    const pageEls = document.querySelectorAll('.slidev-page')
    let currentIdx = 0
    pageEls.forEach((el, idx) => {
      if ((el as HTMLElement).style.display !== 'none') currentIdx = idx + 1
    })
    if (currentIdx > 0 && currentIdx <= totalPages.value) {
      visitedPages.value.add(currentIdx)
      persist()
    }
  })
}

function persist() {
  localStorage.setItem(storageKey, JSON.stringify({
    visitedPages: [...visitedPages.value],
    completedTests: [...completedTests.value],
  }))
}

const pageProgress = () => {
  return totalPages.value > 0
    ? Math.round((visitedPages.value.size / totalPages.value) * 100)
    : 0
}
</script>

<template>
  <div class="progress-widget fixed bottom-4 right-4 z-50">
    <div
      v-if="!expanded"
      class="w-12 h-12 rounded-full bg-gray-800 border border-gray-600 flex items-center justify-center cursor-pointer shadow-lg hover:bg-gray-700 transition-colors"
      @click="expanded = true"
    >
      <svg class="w-5 h-5 text-gray-300" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M9 11l3 3L22 4"/>
        <path d="M21 12v7a2 2 0 01-2 2H5a2 2 0 01-2-2V5a2 2 0 012-2h11"/>
      </svg>
    </div>
    <div
      v-else
      class="bg-gray-800 border border-gray-600 rounded-lg p-3 shadow-lg text-sm min-w-[180px]"
    >
      <div class="flex items-center justify-between mb-2">
        <span class="font-medium text-gray-200">学习进度</span>
        <button class="text-gray-500 hover:text-gray-300" @click="expanded = false">✕</button>
      </div>
      <div class="mb-1.5">
        <div class="flex justify-between text-xs text-gray-400 mb-0.5">
          <span>页面进度</span>
          <span>{{ visitedPages.size }} / {{ totalPages }}</span>
        </div>
        <div class="h-1.5 bg-gray-700 rounded-full overflow-hidden">
          <div
            class="h-full bg-blue-500 rounded-full transition-all duration-300"
            :style="{ width: pageProgress() + '%' }"
          />
        </div>
      </div>
      <div class="text-xs text-gray-500 mt-2">
        数据保存在本地浏览器中
      </div>
    </div>
  </div>
</template>
