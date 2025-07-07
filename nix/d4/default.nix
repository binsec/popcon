{ stdenv, gmp, zlib, boost }:
let
  sources = import ../sources.nix {};
  mkD4 = patches: name: stdenv.mkDerivation {
    pname = name;
    version = sources.d4.rev;
    src = sources.d4;
    buildInputs = [ gmp zlib boost ];
    inherit patches;
    enableParallelBuild = true;
    installPhase = ''
      install -D d4 $out/bin/${name}
    '';
  };
in
{
  d4 = mkD4 [ ./nvars.patch ] "d4";
  d4prio = mkD4 [ ./prio.patch ] "d4prio";
}
