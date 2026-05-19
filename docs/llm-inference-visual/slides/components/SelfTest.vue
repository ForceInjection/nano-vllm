<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'

const props = defineProps<{
  id: string
  question: string
  answer?: string
  options?: string[]
  correct?: number | number[]
  type?: 'text' | 'single' | 'multi'
}>()

const storageKey = computed(() => `nanovllm-selftest-${props.id}`)
const revealed = ref(false)
const selected = ref<number[]>([])
const submitted = ref(false)

onMounted(() => {
  const saved = localStorage.getItem(storageKey.value)
  if (saved) {
    const data = JSON.parse(saved)
    revealed.value = data.revealed || false
    selected.value = data.selected || []
    submitted.value = data.submitted || false
  }
})

function save() {
  localStorage.setItem(storageKey.value, JSON.stringify({
    revealed: revealed.value,
    selected: selected.value,
    submitted: submitted.value,
  }))
}

function toggleReveal() {
  revealed.value = !revealed.value
  save()
}

function toggleOption(index: number) {
  if (props.type === 'single') {
    selected.value = [index]
  } else if (props.type === 'multi') {
    const idx = selected.value.indexOf(index)
    if (idx >= 0) {
      selected.value.splice(idx, 1)
    } else {
      selected.value.push(index)
    }
  }
}

function submitChoice() {
  submitted.value = true
  save()
}

function isCorrect(index: number): boolean {
  if (!submitted.value) return false
  if (Array.isArray(props.correct)) {
    return props.correct.includes(index)
  }
  return props.correct === index
}

function isWrong(index: number): boolean {
  if (!submitted.value) return false
  return selected.value.includes(index) && !isCorrect(index)
}
</script>

<template>
  <div class="selftest border border-gray-600 rounded-lg p-4 my-4">
    <p class="font-semibold text-lg mb-3">{{ question }}</p>

    <!-- Choice mode -->
    <div v-if="type === 'single' || type === 'multi'" class="mb-3">
      <div
        v-for="(opt, idx) in options"
        :key="idx"
        class="choice-item flex items-center gap-2 p-2 rounded cursor-pointer border mb-1"
        :class="{
          'border-gray-500': !submitted,
          'border-green-500 bg-green-500/10': isCorrect(idx),
          'border-red-500 bg-red-500/10': isWrong(idx),
          'opacity-50': submitted && !isCorrect(idx) && !isWrong(idx),
        }"
        @click="!submitted && toggleOption(idx)"
      >
        <span v-if="type === 'multi'" class="text-sm w-5">
          {{ selected.includes(idx) ? '☑' : '☐' }}
        </span>
        <span v-else class="text-sm w-5">
          {{ selected.includes(idx) ? '●' : '○' }}
        </span>
        <span>{{ opt }}</span>
        <span v-if="isCorrect(idx)" class="ml-auto text-green-400">✓</span>
        <span v-if="isWrong(idx)" class="ml-auto text-red-400">✗</span>
      </div>
      <button
        v-if="!submitted && selected.length > 0"
        class="mt-2 px-4 py-1 bg-blue-600 rounded text-sm hover:bg-blue-500"
        @click="submitChoice"
      >
        提交
      </button>
      <p v-if="submitted && answer" class="mt-2 text-sm text-gray-400">
        正确答案：{{ answer }}
      </p>
    </div>

    <!-- Text mode -->
    <div v-if="type === 'text' || !type">
      <button
        class="px-4 py-1.5 rounded text-sm font-medium transition-colors"
        :class="revealed ? 'bg-gray-700 hover:bg-gray-600' : 'bg-blue-600 hover:bg-blue-500'"
        @click="toggleReveal"
      >
        {{ revealed ? '隐藏答案' : '查看答案' }}
      </button>
      <div v-if="revealed && answer" class="mt-3 p-3 bg-gray-800 rounded text-sm leading-relaxed" v-html="answer" />
    </div>
  </div>
</template>

<style scoped>
.choice-item {
  transition: all 0.2s;
}
.choice-item:not(.opacity-50):hover {
  background: rgba(255, 255, 255, 0.05);
}
</style>
