// The `espeak-ng` npm package (1.0.2) ships no types. Only what fonemas.ts uses.
declare module "espeak-ng" {
  interface EspeakModule {
    FS: {
      writeFile(ruta: string, datos: string): void;
      readFile(ruta: string, opts: { encoding: "utf8" }): string;
    };
  }
  interface EspeakOpciones {
    arguments?: string[];
    print?: (s: string) => void;
    printErr?: (s: string) => void;
    preRun?: Array<(m: EspeakModule) => void>;
    noInitialRun?: boolean;
    instantiateWasm?: (
      imports: WebAssembly.Imports,
      cb: (inst: WebAssembly.Instance) => void,
    ) => Record<string, never>;
  }
  export default function ESpeakNg(opts?: EspeakOpciones): Promise<EspeakModule>;
}
