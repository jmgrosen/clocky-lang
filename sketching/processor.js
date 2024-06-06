class ClockyStreamProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.instance = null;
    this.stream = null;
    this.sample_out = null;
    this.first = null;
    console.log("created");

    this.port.onmessage = (event) => {
      console.log("messaged");
      this.instance = null;
      WebAssembly.instantiate(event.data.clockyModule).then(({instance}) => {
        console.log("instantiated");
        this.instance = instance;
        this.first = true;
        instance.exports.init_scheduler()
        const clock0 = instance.exports.get_clock_data(0);
        const intermediate = instance.exports.apply_clock(instance.exports.main, clock0);
        const clock1 = instance.exports.get_clock_data(1);
        this.stream = instance.exports.apply_clock(intermediate, clock1);
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

      this.stream = this.instance.exports.sample_scheduler(this.stream, 128, this.samples_ptr);
      const samples = new Float32Array(this.instance.exports.memory.buffer, this.samples_ptr, 128);

      const output = outputs[0];
      output.forEach((channel) => {
        channel.set(samples, 0);
      });
    }
    return true;
  }
}

registerProcessor("clocky-stream-processor", ClockyStreamProcessor);
