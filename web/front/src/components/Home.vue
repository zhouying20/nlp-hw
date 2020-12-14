<template>
  <el-container id="event-extractor">
    <el-header style="text-align: left; font-size: 40px" class="padding">
      <span>Event Extraction</span>
    </el-header>

    <el-container direction="vertical">
      <!-- set progressbar -->
      <vue-progress-bar></vue-progress-bar>
      <el-main class="padding" style="text-align: left;">
        <span>
          Event extraction is a long-studied and challenging task in Information Extraction (IE). The goal of event extraction is to detect event instance(s) in texts, and if existing, identify the event type as well as all of its participants and attributes.
        </span>
        <div style="margin: 20px 0 20px 0">
          Model:
          <el-select style="margin-left: 10px; width:60%" size ="medium" v-model="selected_model" placeholder="Model">
            <el-option-group
              v-for="group in model_types"
              :key="group.label"
              :label="group.label">
              <el-option
                v-for="item in group.models"
                :key="item.value"
                :label="item.label"
                :value="item.value">
              </el-option>
            </el-option-group>
          </el-select>
        </div>
        <span>
          {{description}}
        </span>
      </el-main>

      <el-divider><i class="el-icon-news"></i></el-divider>

      <el-container class="padding" direction="vertical">
        <el-main>
          <div style="margin-bottom: 30px">
            <span style="color: gray">Enter text or </span>
            <el-select style="margin-left: 5px; width:58%" size ="medium" v-model="selected_text" placeholder="Choose an example.">
              <el-option-group
                v-for="group in examples"
                :key="group.label"
                :label="group.label">
                <el-option
                  v-for="item in group.texts"
                  :key="item.text"
                  :label="item.text"
                  :value="item.text">
                </el-option>
              </el-option-group>
            </el-select>
          </div>
          <span>Sentence or Document</span>
          <el-input
            type="textarea"
            :rows="4"
            autofocus
            placeholder="Sentence or Document to extract."
            style="margin-top: 10px"
            v-model="text">
          </el-input>
        </el-main>
        <el-main style="text-align: center;">
          <el-button 
          round="true" 
          plain="true" 
          type="primary" 
          icon="el-icon-arrow-right"
          @click="infer"
          :loading="inferring">
          Extract !
          </el-button>
        </el-main>
      </el-container>

      <el-divider></el-divider>

      <el-container v-loading="inferring" direction="vertical">
        <el-main class="padding" style="width=60%">
          <div style="margin-bottom=30px">
            Each color represents: 
            <el-tag 
            type="danger"
            style="margin: 2px 0 20px 4px">
            Event Trigger
            </el-tag>
            <el-tag 
            style="margin: 2px 0 20px 4px">
            Argument
            </el-tag>
          </div>
          <el-tag 
          v-for="(token, index) in tokens" 
          :key="index" 
          :type="token[1]==0 ? 'info' : token[1]==1 ? 'danger' : '' "
          style="margin: 2px 0 2px 4px">
          {{token[0]}}
          </el-tag>
        </el-main>

        <el-main v-if="events" class="padding">
         <div v-for="event in events" :key="event">
           {{event.type}} : {{event.token}}
           <el-table
            :data="event.arguments"
            style="margin-bottom: 40px">
            <el-table-column
              prop="type"
              label="Argument Role"
              width="180">
            </el-table-column>
            <el-table-column
              prop="token"
              label="Tokens"
              width="180">
            </el-table-column>
          </el-table>
         </div>
        </el-main>
      </el-container>

    </el-container>
  </el-container>
</template>

<script>
import axios from 'axios'

export default {
  data () {
    return {
      infer_apis: [
        '/api/seqseq',
        '/api/seqqa',
        '/api/qaseq',
        '/api/qaqa',
        // 'http://124.16.71.41:5000/eeqa/infer',
      ],
      model_types: [
        {
          'label': 'Sentence',
          'models': [
            {
              'label': 'Our Method',
              'value': 0
            },
            // {
            //   'label': 'Our Method 2 seqqa',
            //   'value': 1
            // },
            // {
            //   'label': 'Our Method 3 qaseq',
            //   'value': 2
            // },
            {
              'label': 'Event Extraction by Answering (Almost) Natural Questions',
              'value': 3
            }
          ]
        },
        // {
        //   'label': 'Document',
        //   'models': [
        //     {
        //       'label': 'Document-Level Event Role Filler Extraction using Multi-Granularity Contextualized Encoding',
        //       'value': 1
        //     }
        //   ]
        // }
      ],
      descriptions: [
        'We formalize EE as a sequence labeling task, and implement a LSTM-CRF model to predict the final result. The model uses the context representation generated by BERT as input. The model was trained on the ACE2005 dataset, achieved better performance than DMCNN.',
        'We formalize EE as a sequence labeling task, and implement a LSTM-CRF model to predict the final result. The model uses the context representation generated by BERT as input. The model was trained on the ACE2005 dataset, achieved better performance than DMCNN.',
        'We formalize EE as a sequence labeling task, and implement a LSTM-CRF model to predict the final result. The model uses the context representation generated by BERT as input. The model was trained on the ACE2005 dataset, achieved better performance than DMCNN.',
        'This model is described in Xinya Du and Claire Cardie 2020 EMNLP. It formalize EE as a question answering (QA) task, and it starts with pretrained BERT as base model for obtaining contextualized representations from the input sequences. It was trained on the ACE2005 dataset.',
        // 'This model is described in Xinya Du and Claire Cardie 2020 ACL. It formalize Document level EE as a sequence tagging task over the tokens in a set of contiguous sentences in the document, and it starts with pretrained BERT for obtaining contextualized representations, then use LSTM and CRF for sequence tagging. It was trained on the MUC-4 dataset.'
      ],
      examples: [
        {
          'label': 'Sentence',
          'texts': [
            {
              'text': 'The Belgrade district court said that Markovic will be tried along with 10 other Milosevic-era officials who face similar charges of "inappropriate use of state property" that carry a sentence of up to five years in jail.'
            },
            {
              'text': "Police have arrested four people in connection with the killings."
            },
            {
              'text': "Prison authorities have given the nod for Anwar to be taken home later in the afternoon to marry his eldest daughter, Nurul Izzah, to engineer Raja Ahmad Sharir Iskandar in a traditional Malay ceremony, he said."
            },
            {
              'text': "Kristin Scott, the mother, told police she gave birth secretly to both babies at her parents' home in Byrds Creek, Richland County, one of unknown sex in April 2001 and the other, a fullterm girl, January 14."
            },
            {
              'text': "She is being held on 50,000 dollars bail on a charge of first-degree reckless homicide and hiding a corpse in the death of the infant born in January."
            },
            {
              'text': "British Chancellor of the Exchequer Gordon Brown on Tuesday named the current head of the country's energy regulator as the new chairman of finance watchdog the Financial Services Authority(FSA)."
            },
          ]
        },
        // {
        //   'label': 'Document',
        //   'texts': [
        //     {
        //       'shorted': 'Test Document 1',
        //       'text': 'Test Document 1'
        //     },
        //     {
        //       'shorted': 'Test Document 2',
        //       'text': 'Test Document 2'
        //     },
        //   ]
        // },
      ],
      selected_model: 0,
      selected_text: '',
      text: '',
      // tokens: [["Russian",0],["President",0],["Vladimir",2],["Putin",2],["'s",0],["summit",1],["with",0],["the",0],["leaders",2],["of",0],["Germany",0],["and",0],["France",0],["may",0],["have",0],["been",0],["a",0],["failure",0],["that",0],["proves",0],["there",0],["can",0],["be",0],["no",0],["long",0],["-",0],["term",0],["\"",0],["peace",0],["camp",0],["\"",0],["alliance",0],["following",0],["the",0],["end",0],["of",0],["war",1],["in",0],["Iraq",2],[",",0],["government",0],["sources",0],["were",0],["quoted",0],["as",0],["saying",0],["at",0],["the",0],["weekend",0],[".",0]],
      // events: [{"arguments":[{"token":"Vladimir Putin","type":"Contact.Meet_Entity"},{"token":"leaders","type":"Contact.Meet_Entity"}],"token":"summit","type":"Contact.Meet"},{"arguments":[{"token":"Iraq","type":"Conflict.Attack_Place"}],"token":"war","type":"Conflict.Attack"}],
      tokens: [],
      events: [],
      inferring: false,
    }
  },
  computed: {
    description: function () {
      return this.descriptions[this.selected_model]
    },
  },
  watch: {
    selected_text: function (val, oldVal) {
      this.text = val;
    }
  },
  methods: {
    // update_events () {
    //   this.events_list = []
    //   for (var k in this.events) {
    //     var events_dict = {}
    //     events_dict['event_type'] = k
    //     var trigger_offset = this.events[k]['offset']
    //     events_dict['trigger_word'] = this.tokens[trigger_offset]

    //     var one_table = []
    //     for (var args in this.events[k]['arguments']) {
    //       one_table.push({
    //         'argument_type': args[0],
    //         'start_token': this.tokens[args[1][0]],
    //         'end_token': this.tokens[args[1][1]],
    //       })
    //     }
    //     events_dict['arguments'] = one_table
    //     this.events_list.push(events_dict)
    //   }
    // },
    infer () {
      // infer_url = this.infer_apis[this.selected_model]
      this.inferring = true
      axios({ 
        method: 'POST', 
        url: this.infer_apis[this.selected_model],
        data: {
          'sentence': this.text
        }
      }).then(
        result => {
          this.tokens = result.data.tokens
          this.events = result.data.events
          // this.update_events()
          this.inferring = false
        },
        error => {
          console.error(error)
          this.inferring = false
        }
      )
    },
  }
}
</script>

<style>
  .el-header {
    /* background-color: #409EFF; */
    color: #333;
    line-height: 60px;
  }
  
  .el-aside {
    color: #333;
  }

  .padding {
    margin-left: 10%;
    margin-right: 10%;
  }

  .el-select-dropdown{
    max-width: 243px;
  }
  .el-select-dropdown__item{
    display: block;
    overflow-wrap: break-word;
  }
  .el-select-dropdown__item span {
    display: block;
  }
</style>