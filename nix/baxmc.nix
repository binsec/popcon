{stdenv}:
    let sources = import ./sources.nix {}; in
    (import sources.flake-compat {src = sources.baxmc;}).outputs.packages.x86_64-linux.default
