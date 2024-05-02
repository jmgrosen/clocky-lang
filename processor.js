/*
let stream = instance.exports.main;
const num_samples = 1000;
const samples_ptr = instance.exports.sample(stream, num_samples);
const mem_f32 = new Float32Array(instance.exports.memory.buffer);
const mem_i32 = new Uint32Array(instance.exports.memory.buffer);
const samples = mem_f32.slice(samples_ptr/4, samples_ptr/4 + num_samples);
console.log(samples);
*/

class ClockyStreamProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.instance = null;
    this.stream = null;
    this.sample_out = null;
    this.first = true;
    console.log("created");

    this.port.onmessage = (event) => {
      console.log("messaged");
      WebAssembly.instantiate(event.data.clockyModule).then(({instance}) => {
        console.log("instantiated");
        this.instance = instance;
        this.stream = instance.exports.main;
        this.samples_ptr = instance.exports.alloc(4 * 128);
      });
    };
  }

  process(inputs, outputs, parameters) {
    if (this.instance) {
      if (this.first) {
        this.port.postMessage({goodToGo: true});
        this.first = false;
      }

      this.stream = this.instance.exports.sample(this.stream, 128, this.samples_ptr);
      const mem_f32 = new Float32Array(this.instance.exports.memory.buffer);
      const samples = mem_f32.slice(this.samples_ptr/4, this.samples_ptr/4 + 128);

      const output = outputs[0];
      output.forEach((channel) => {
        for (let i = 0; i < channel.length; i++) {
          channel.set(samples, 0);
        }
      });
    }
    return true;
  }
}

registerProcessor("clocky-stream-processor", ClockyStreamProcessor);
