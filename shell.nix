let
  pkgs = import ./nix/pkgs.nix;
  sources = import ./nix/sources.nix {};
in
pkgs.mkShell {
  buildInputs = with pkgs; [
    boolector
    c2d
    aiger
    rustc
    cargo
    cryptominisat
    rustfmt
    clippy
    graph-easy
    dsharp
    crate2nix
    maxcount
    cargo-outdated
    creduce
    query-dnnf
    rust-analyzer
    ssatABC
    d4
    d4prio
    clasp
    dc-ssat
    baxmc
    (python3.withPackages (ps: with ps; [ python-prctl pandas matplotlib ]))
  ];
  RUST_BACKTRACE = "1";
}
