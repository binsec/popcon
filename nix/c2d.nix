{ runCommand, fetchzip, lib }:
let
  mc2020 = fetchzip {
    url = "https://zenodo.org/record/4292581/files/MC2020_Solvers.tar.bz2";
    sha256 = "017p66la4lldbxqm3gg6pha7mpfibjw8ydjdnrgkdn1hrchxgwji";
  };
in
  runCommand "c2d-mc2020" {
    meta.license = lib.licenses.unfree;
  } ''
    install -Dt $out/bin/ ${mc2020}/track1/c2d-mc-solver/bin/c2d
  ''

