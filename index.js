const clockyModulePromise = fetch("test.wasm");

window.go = async () => {
  const audioContext = new AudioContext();
  await audioContext.audioWorklet.addModule("processor.js");
  const clockyNode = new AudioWorkletNode(audioContext, "clocky-stream-processor");
  const clockyModule = await (await clockyModulePromise).arrayBuffer();
  console.log("awaited");
  clockyNode.port.postMessage({clockyModule: clockyModule});
  clockyNode.connect(audioContext.destination);
};
