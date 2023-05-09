extern crate proc_macro;
use proc_macro::TokenStream;
use syn::{parse_macro_input, DataEnum, DataUnion, DeriveInput, FieldsNamed, FieldsUnnamed};

#[proc_macro_derive(AnswerFn)]
pub fn derive_answer_fn(item: TokenStream) -> TokenStream {
    let DeriveInput { ident, data, .. } = parse_macro_input!(item);
    println!("item: \"{}\"", ident.to_string());

    "fn answer() -> u32 { 42 }".parse().unwrap()
}
