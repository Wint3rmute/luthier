extern crate proc_macro;
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    parse_macro_input, punctuated::Punctuated, token::Comma, DeriveInput, Field, FieldsNamed,
    FieldsUnnamed,
};

fn find_input_fields(fields: &Punctuated<Field, Comma>) -> Vec<String> {
    let mut result = vec![];
    for field in fields.iter() {
        if field
            .ident
            .clone()
            .unwrap()
            .to_string()
            .starts_with("input_")
        {
            result.push(field.ident.clone().unwrap().to_string())
        }
    }

    result
}

fn find_output_fields(fields: &Punctuated<Field, Comma>) -> Vec<String> {
    let mut result = vec![];
    for field in fields.iter() {
        if field
            .ident
            .clone()
            .unwrap()
            .to_string()
            .starts_with("output_")
        {
            result.push(field.ident.clone().unwrap().to_string())
        }
    }

    result.sort();
    result
}

#[proc_macro_derive(DspConnectibleDerive)]
pub fn derive_answer_fn(input: TokenStream) -> TokenStream {
    let DeriveInput { ident, data, .. } = parse_macro_input!(input);

    let (inputs, outputs) = match data {
        syn::Data::Struct(s) => match s.fields {
            syn::Fields::Named(FieldsNamed { named, .. }) => {
                let inputs = find_input_fields(&named);
                let outputs = find_output_fields(&named);
                (inputs, outputs)
            }
            syn::Fields::Unnamed(FieldsUnnamed { unnamed: _, .. }) => {
                panic!("Unnamed fields not supported");
            }
            syn::Fields::Unit => panic!("Unnamed fields not supported"),
        },
        _ => {
            panic!("Only structs are supported as macro inputs")
        }
    };

    let mut input_indexes = vec![];
    let mut input_idents = vec![];
    for i in 0..inputs.len() {
        input_indexes.push(i);
        input_idents.push(format_ident!("{}", inputs[i]));
    }

    let inputs_enum_ident = format_ident!("{}Inputs", ident);
    let outputs_enum_ident = format_ident!("{}Outputs", ident);
    let inputs_enum_values: Vec<syn::Ident> = inputs
        .iter()
        .map(|v| format_ident!("{}", v.to_uppercase()))
        .collect();
    let outputs_enum_values: Vec<syn::Ident> = outputs
        .iter()
        .map(|v| format_ident!("{}", v.to_uppercase()))
        .collect();

    let mut output_indexes = vec![];
    let mut output_idents = vec![];
    for i in 0..outputs.len() {
        output_indexes.push(i);
        output_idents.push(format_ident!("{}", outputs[i]));
    }

    let mut output = quote! {
    impl DspConnectible for #ident {
        fn node_name(&self) -> &str {
            stringify!(#ident)
        }

        fn get_input_names(&self) -> Vec<String> {
            vec![
                #(#inputs.to_string(),)*
            ]
        }

        fn get_output_names(&self) -> Vec<String> {
            vec![
                #(#outputs.to_string(),)*
            ]
        }

        fn get_output_by_index(&self, index: usize) -> f64 {
            match index {
                #( #output_indexes => self.#output_idents , )*
                _ => panic!("Output {} not found in {}", index, stringify!(#ident))
            }
        }

        fn get_input_by_index(&self, index: usize) -> f64 {
            match index {
                #( #input_indexes => self.#input_idents , )*
                _ => panic!("Input {} not found in {}", index, stringify!(#ident))
            }
        }

        fn set_input_by_index(&mut self, index: usize, value: f64) {
            match index {
                #( #input_indexes => self.#input_idents = value, )*
                _ => panic!("Input {} not found in {}", index, stringify!(#ident))
            }
        }
    }
    };

    if inputs_enum_values.len() > 0 {
        output.extend(quote! {

        #[derive(Debug, Clone)]
        #[repr(usize)]
        enum #inputs_enum_ident {
            #( #inputs_enum_values = #input_indexes ,)*
        }
        });
    }

    if outputs_enum_values.len() > 0 {
        output.extend(quote! {

        #[derive(Debug, Clone)]
        #[repr(usize)]
        enum #outputs_enum_ident {
            #( #outputs_enum_values = #output_indexes ,)*
        }
        });
    }
    output.into()
}
