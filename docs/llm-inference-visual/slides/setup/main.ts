import { defineAppSetup } from '@slidev/types'
import SelfTest from '../components/SelfTest.vue'
import SourceCode from '../components/SourceCode.vue'
import ProgressWidget from '../components/ProgressWidget.vue'
import '../style.css'

export default defineAppSetup(({ app }) => {
  app.component('SelfTest', SelfTest)
  app.component('SourceCode', SourceCode)
  app.component('ProgressWidget', ProgressWidget)
})
