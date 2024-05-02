const clockyModulePromise = fetch("test.wasm");

window.onload = () => {
  const button = window.document.querySelector("button");
  console.log(button);

  button.addEventListener("click", async (event) => {
    button.innerText = "starting";
    const audioContext = new AudioContext();
    await audioContext.audioWorklet.addModule("processor.js");
    const clockyNode = new AudioWorkletNode(audioContext, "clocky-stream-processor");
    const clockyModule = await (await clockyModulePromise).arrayBuffer();
    console.log("awaited");
    clockyNode.port.onmessage = (event) => {
      console.log("boo");
      button.innerText = "going";
    };
    clockyNode.port.postMessage({clockyModule: clockyModule});
    clockyNode.connect(audioContext.destination);
  });
};
