<script setup lang="ts">
const props = defineProps<{
  file: string
  lines?: string
  repo?: string
}>()

const baseRepo = props.repo || 'GeeeekExplorer/nano-vllm'
const branch = 'main'

import { computed } from 'vue'

const url = computed(() => {
  const linePart = props.lines ? `#L${props.lines.replace('-', '-L')}` : ''
  return `https://github.com/${baseRepo}/blob/${branch}/${props.file}${linePart}`
})

const label = computed(() => {
  const linePart = props.lines ? `:${props.lines}` : ''
  return `${props.file}${linePart}`
})
</script>

<template>
  <div class="source-code-header mb-1 flex items-center text-xs">
    <a :href="url" target="_blank" class="text-blue-400 hover:text-blue-300 transition-colors inline-flex items-center gap-1">
      <svg class="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
        <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"/>
      </svg>
      {{ label }}
    </a>
  </div>
</template>
