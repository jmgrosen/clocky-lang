use std::collections::HashMap;
use std::ops::Range;

use indexmap::IndexSet;
use wasm_encoder as wasm;

pub struct Global {
    pub ty: wasm::GlobalType,
    pub init_expr: wasm::ConstExpr,
}

pub struct Runtime<'a> {
    pub buf: &'a [u8],
    pub exports: HashMap<String, (wasmparser::ExternalKind, u32)>,
    pub types: IndexSet<wasm::FuncType>,
    pub functions: Vec<u32>,
    // tables: _,
    pub initial_memory_size: u32,
    pub code: Vec<Range<usize>>,
    pub data: Vec<wasmparser::Data<'a>>,
    pub globals: Vec<Global>,
}

fn valtype_to_valtype(ty: &wasmparser::ValType) -> wasm_encoder::ValType {
    match *ty {
        wasmparser::ValType::I32 => wasm_encoder::ValType::I32,
        wasmparser::ValType::I64 => wasm_encoder::ValType::I64,
        wasmparser::ValType::F32 => wasm_encoder::ValType::F32,
        wasmparser::ValType::F64 => wasm_encoder::ValType::F64,
        wasmparser::ValType::V128 => wasm_encoder::ValType::V128,
        wasmparser::ValType::Ref(_) => panic!("can't convert reftypes yet >:("),
    }
}

fn globaltype_to_globaltype(ty: &wasmparser::GlobalType) -> wasm::GlobalType {
    wasm::GlobalType {
        val_type: valtype_to_valtype(&ty.content_type),
        mutable: ty.mutable,
        shared: ty.shared,
    }
}

fn functype_to_functype(ty: wasmparser::FuncType) -> wasm::FuncType {
    wasm::FuncType::new(ty.params().iter().map(valtype_to_valtype),
                        ty.results().iter().map(valtype_to_valtype))
}

fn constexpr_to_constexpr(ce: &wasmparser::ConstExpr<'_>) -> wasm::ConstExpr {
    // why isn't this a method??
    let mut reader = ce.get_binary_reader();
    let bytes = reader.read_bytes(reader.bytes_remaining()).unwrap();
    println!("constexpr bytes in: {bytes:?}");
    let constexpr_out = wasm::ConstExpr::raw(bytes.iter().copied().take(bytes.len()-1));
    let mut out_bytes = Vec::new();
    use wasm_encoder::Encode;
    constexpr_out.encode(&mut out_bytes);
    println!("constexpr bytes out: {bytes:?}");
    constexpr_out
}

fn exportkind_to_exportkind(k: wasmparser::ExternalKind) -> wasm::ExportKind {
    match k {
        wasmparser::ExternalKind::Func => wasm::ExportKind::Func,
        wasmparser::ExternalKind::Table => wasm::ExportKind::Table,
        wasmparser::ExternalKind::Memory => wasm::ExportKind::Memory,
        wasmparser::ExternalKind::Global => wasm::ExportKind::Global,
        wasmparser::ExternalKind::Tag => wasm::ExportKind::Tag,
    }
}

impl<'a> Runtime<'a> {
    // panics if buf is invalid
    pub fn from_bytes(buf: &'a [u8]) -> Runtime<'a> {
        use wasmparser::Payload::*;

        let parser = wasmparser::Parser::new(0);
        let mut types = IndexSet::new();
        let mut functions = Vec::with_capacity(0);
        let mut code = Vec::with_capacity(0);
        let mut initial_memory_size = 0;
        let mut globals = Vec::with_capacity(0);
        let mut exports = HashMap::new();
        let mut data = Vec::with_capacity(0);
        for payload in parser.parse_all(buf) {
            match payload.unwrap() {
                Version { .. } => { }
                TypeSection(type_reader) => {
                    for func_type in type_reader.into_iter_err_on_gc_types() {
                        types.insert(functype_to_functype(func_type.unwrap()));
                    }
                }
                ImportSection(_) => { panic!("runtime should not have imports") }
                FunctionSection(func_reader) => {
                    functions = func_reader.into_iter().collect::<Result<_, _>>().unwrap();
                }
                TableSection(table_reader) => {
                    // TODO: make sure we're handling this right. this
                    // will be especially tricky if we define any
                    // vtables within the runtime itself... which we
                    // surely will if we pull in nontrivial libraries
                    assert!(table_reader.count() == 1);
                    let table = table_reader.into_iter().next().unwrap().unwrap();
                    assert!(table.ty.element_type == wasmparser::RefType::FUNCREF);
                    // TODO: is this what we expect? this is just what i found at first.
                    assert!(table.ty.initial == 1);
                    assert!(table.ty.maximum == Some(1));
                }
                MemorySection(mem_reader) => {
                    assert!(mem_reader.count() == 1);
                    let mem = mem_reader.into_iter().next().unwrap().unwrap();
                    initial_memory_size = mem.initial as u32;
                }
                TagSection(_) => { panic!("tag section?") }
                GlobalSection(global_reader) => {
                    globals = global_reader
                        .into_iter()
                        .map(|rg| rg.map(|g| Global {
                            ty: globaltype_to_globaltype(&g.ty),
                            init_expr: constexpr_to_constexpr(&g.init_expr),
                        }))
                        .collect::<Result<_, _>>().unwrap();
                }
                ExportSection(export_reader) => {
                    for export in export_reader {
                        let export = export.unwrap();
                        exports.insert(export.name.to_string(), (export.kind, export.index));
                    }
                }
                StartSection { .. } => { panic!("runtime should not have start, right?") }
                ElementSection(_) => { panic!("element section?") }

                DataCountSection { count, .. } => {
                    assert!(count == 1);
                }
                DataSection(data_reader) => {
                    data = data_reader.into_iter().collect::<Result<_, _>>().unwrap();
                }

                CodeSectionStart { count, .. } => {
                    code.reserve_exact(count as usize);
                }
                CodeSectionEntry(body) => {
                    code.push(body.range());
                }

                CustomSection(_) => { }

                End(_) => { }

                section => { panic!("weird section {section:?}") }
            }
        }

        Runtime {
            buf,
            exports,
            types,
            functions,
            initial_memory_size,
            code,
            data,
            globals,
        }
    }

    pub fn emit_functions(&self, functions: &mut wasm::FunctionSection) {
        for &func in self.functions.iter() {
            functions.function(func);
        }
    }

    pub fn emit_globals(&self, globals: &mut wasm::GlobalSection) {
        for g in self.globals.iter() {
            globals.global(g.ty, &g.init_expr);
        }
    }

    pub fn emit_code(&self, codes: &mut wasm::CodeSection) {
        for body in self.code.iter() {
            codes.raw(&self.buf[body.clone()]);
        }
    }

    pub fn emit_data(&self, data: &mut wasm::DataSection) {
        for segment in self.data.iter() {
            match segment.kind {
                wasmparser::DataKind::Passive => {
                    data.passive(segment.data.iter().copied());
                }
                wasmparser::DataKind::Active { memory_index, ref offset_expr } => {
                    data.active(memory_index, &constexpr_to_constexpr(offset_expr), segment.data.iter().copied());
                }
            }
        }
    }

    pub fn emit_exports(&self, exports: &mut wasm::ExportSection) {
        for (name, &(k, i)) in self.exports.iter() {
            exports.export(name, exportkind_to_exportkind(k), i);
        }
    }
}
