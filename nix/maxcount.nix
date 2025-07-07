{ python3, cryptominisat, stdenv, makeWrapper, coreutils, lib }:
let sources = import ./sources.nix {}; in
stdenv.mkDerivation {
  pname = "maxcount";
  version = sources.rev or "unknown";

  src = sources.maxcount;

  patches = [ ./maxcount-prctl.patch ];

  nativeBuildInputs = [ makeWrapper ];
  installPhase = ''
    mkdir -p $out/bin $out/opt
    mv *.py LICENSE $out/opt
    mv scalmc-binaries/x86_64-linux/scalmc $out/bin
    substituteInPlace $out/opt/maxcount.py --replace selfcomposition.py $out/opt/selfcomposition.py
    makeWrapper ${python3.interpreter} $out/bin/maxcount \
    --add-flags "$out/opt/maxcount.py --scalmc $out/bin/scalmc" \
    --prefix PYTHONPATH : ${lib.makeSearchPath python3.sitePackages [ cryptominisat python3.pkgs.python-prctl ]} \
    --prefix PATH : ${lib.makeBinPath [ python3 coreutils ]}
  '';

  doInstallCheck = true;
  installCheckPhase = ''
    PYTHONPATH= $out/bin/maxcount --help
  '';
}
