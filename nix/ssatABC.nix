{ stdenv, readline, zlib, gmp, enableDebugging }:
let
  sources = import ./sources.nix {};
  src = sources.ssatABC;
in
stdenv.mkDerivation {
  pname = "ssatABC";
  version = src.rev or "local";

  patches = [ ./ssatABC_gmp.patch ./ssatABC_loop.patch ];

  enableParallelBuilding = true;

  inherit src;

  buildInputs = [ readline zlib (enableDebugging gmp) ];

  dontStrip = true;

  installPhase = ''
    install -Dt $out/bin bin/abc
  '';
}




