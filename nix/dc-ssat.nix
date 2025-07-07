{ stdenv }:
let sources = import ./sources.nix {}; in
stdenv.mkDerivation rec {
  pname = "dc-ssat";
  version = src.rev or "local";
  src = sources.pldi.outPath + "/DC-SSAT";

  installPhase = ''
    install -Dt $out/bin dcssat
  '';
}
