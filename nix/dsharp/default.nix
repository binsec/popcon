{ stdenv, gmp, }:
let
  sources = import ../sources.nix {};
in
stdenv.mkDerivation {
  pname = "dsharp";
  version = sources.dsharp.rev;
  src = sources.dsharp;
  patches = [ ./unsat_nvars.patch ];
  buildInputs = [ gmp ];
  preConfigure = ''
    cp Makefile_gmp Makefile
  '';
  installPhase = ''
    install -Dt $out/bin dsharp
  '';
}
