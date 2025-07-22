let src_lib = import "src/lib.rs" as 'Text in
let src_builder = import "src/builder.rs" as 'Text in
let src_signal_vec = import "src/signal_vec.rs" as 'Text in
let src_signal = import "src/signal.rs" as 'Text in
let src_signal_map = import "src/signal_map.rs" as 'Text in
let src_graph = import "src/graph.rs" as 'Text in
let src_utils = import "src/utils.rs" as 'Text in
let examples_basic_builder = import "examples/basic_builder.rs" as 'Text in
let examples_basic = import "examples/basic.rs" as 'Text in
let examples_lifetime_list = import "examples/lifetime_list.rs" as 'Text in
let examples_filters = import "examples/filters.rs" as 'Text in
let examples_test = import "examples/test.rs" as 'Text in
let examples_utils = import "examples/utils.rs" as 'Text in
let repo_structure = {
  "src/lib.rs" = src_lib,
  "src/builder.rs" = src_builder,
  "src/signal_vec.rs" = src_signal_vec,
  "src/signal.rs" = src_signal,
  "src/signal_map.rs" = src_signal_map,
  "src/graph.rs" = src_graph,
  "src/utils.rs" = src_utils,
  "examples/basic_builder.rs" = examples_basic_builder,
  "examples/basic.rs" = examples_basic,
  "examples/lifetime_list.rs" = examples_lifetime_list,
  "examples/filters.rs" = examples_filters,
  "examples/test.rs" = examples_test,
  "examples/utils.rs" = examples_utils,
} in
let prompt = import "repo_prompt_input.md" as 'Text in

m%"
%{ prompt }

jonmo is a functional reactive signals implementation in rust for the bevy game engine; here is the structure of the jonmo repository with the contents of each file:
.
├── examples/
│   ├── basic_builder.rs
│   ├── basic.rs
│   ├── lifetime_list.rs
│   ├── filters.rs
│   ├── test.rs
│   └── utils.rs
└── src/
    ├── builder.rs
    ├── lib.rs
    ├── signal_vec.rs
    ├── signal.rs
    ├── signal_map.rs
    ├── graph.rs
    └── utils.rs

%{ repo_structure |> std.record.map (fun k v => "%{ k }\n```rust\n%{ v }\n```") |> std.record.values |> std.string.join "\n\n" }
"%
