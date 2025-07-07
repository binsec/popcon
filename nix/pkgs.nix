let
  sources = import ./sources.nix { };
  pkgs = import sources.nixpkgs {
    config.allowUnfree = true;
    overlays = [ (self: super: {
      dsharp = super.callPackage ./dsharp {};
      c2d = super.callPackage ./c2d.nix {};
      maxcount = super.callPackage ./maxcount.nix {};
      crate2nix = super.callPackage sources.crate2nix {};
      ssatABC = super.callPackage ./ssatABC.nix {};
      dc-ssat = super.enableDebugging (super.callPackage ./dc-ssat.nix {});
      inherit (super.callPackages ./d4 {}) d4 d4prio;
      query-dnnf = super.callPackage ./query-dnnf.nix {};
      baxmc = super.callPackage ./baxmc.nix {};
      popcon = super.callPackage ./popcon.nix {};
    }) ];
  };
in
  pkgs
