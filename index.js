import init, { compile } from "./pkg/clocky.js";

window.onload = async () => {
  await init();

  const button = window.document.querySelector("button");
  const status = window.document.querySelector("#status");
  const codeArea = window.document.querySelector("textarea");

  const audioContext = new AudioContext();
  await audioContext.audioWorklet.addModule("processor.js");
  const clockyNode = new AudioWorkletNode(audioContext, "clocky-stream-processor");
  clockyNode.port.onmessage = (event) => {
    console.log("boo");
    status.innerText = "running";
  };
  clockyNode.connect(audioContext.destination);

  button.addEventListener("click", async (event) => {
    console.log(codeArea.value);
    status.innerText = "compiling";
    const clockyModule = compile(codeArea.value);
    status.innerText = "sending";
    clockyNode.port.postMessage({clockyModule});
    if (audioContext.state === "suspended") {
      audioContext.resume();
    }
  });
};
