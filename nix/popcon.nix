{ lib, c2d, d4, d4prio, dsharp, makeWrapper, maxcount, boolector, pkgs, ssatABC, dc-ssat , baxmc}:
let
  customBuildRustCrateForPkgs = pkgs: pkgs.buildRustCrate.override {
    defaultCrateOverrides = pkgs.defaultCrateOverrides // {
      popcon = attrs: {
        dontStrip = true;
        nativeBuildInputs = [ makeWrapper ];
        postInstall = ''
          # skip this when there is no $out/bin/popcon, as is the case in the derivation running tests
          if [ -x $out/bin/popcon ]; then
            wrapProgram $out/bin/popcon --prefix PATH : ${lib.makeBinPath [ c2d dsharp boolector d4 d4prio ssatABC maxcount dc-ssat baxmc ]}
          fi
        '';
      };
    };
  };
  generatedBuild = import ./Cargo.nix {
    inherit pkgs;
    buildRustCrateForPkgs = customBuildRustCrateForPkgs;
  };
in generatedBuild.rootCrate.build.override {
    runTests = true;
    testInputs = [ c2d dsharp maxcount boolector d4 d4prio ssatABC dc-ssat baxmc];
  }
