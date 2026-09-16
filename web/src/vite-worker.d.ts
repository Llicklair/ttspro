/**
 * El import de worker de Vite (`?worker`), que TypeScript no conoce solo.
 *
 * `vite/client` traería esto y mucho más, pero también `import.meta.env` y el
 * resto del ambiente del bundler, que este proyecto no usa en ningún sitio. Se
 * declara lo único que hace falta.
 */
declare module "*?worker" {
  const Trabajador: new () => Worker;
  export default Trabajador;
}
