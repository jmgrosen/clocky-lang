import { defaultKeymap, history, historyKeymap } from "@codemirror/commands";
import { EditorView, keymap, lineNumbers } from "@codemirror/view";
import { setup } from "sketchzone";
import { compile } from "../pkg";
import processorUrl from "./processor.js?worker&url";

import exampleWaves from "../tests/accept/waves.cky?raw";
import example440 from "../tests/accept/440.cky?raw";

function createAndMountInspector(root, doc) {
  try {
    const clockyModule = compile(doc);
    root.innerText = "sending";
    window.clockyNode.port.postMessage({ clockyModule });
    if (window.audioContext.state === "suspended") {
      window.audioContext.resume();
    }
  } catch (e) {
    root.innerText = e;
  }
}

const audioContext = new AudioContext();
await audioContext.audioWorklet.addModule(processorUrl);
window.audioContext = audioContext;
const clockyNode = new AudioWorkletNode(
  audioContext,
  "clocky-stream-processor"
);
window.clockyNode = clockyNode;
clockyNode.connect(audioContext.destination);

setup({
  createAndMountInspector,
  appName: "clocky",
  infoUrl: "https://github.com/jmgrosen/clocky-lang",
  codemirrorExtensions: [
    lineNumbers(),
    history(),
    EditorView.lineWrapping,
    keymap.of([...defaultKeymap, ...historyKeymap]),
  ],
  defaultEntries: [
    exampleWaves,
    example440,
  ],
  extractTitleFromDoc: (doc) => {
    if (!doc.startsWith("def ")) {
      return "[unnamed]";
    } else {
      return doc.slice(4).split(" ")[0]; // TODO: regex
    }
  },
});
