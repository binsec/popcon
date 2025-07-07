{ gmp, stdenv, fetchurl, unzip }:
stdenv.mkDerivation {
  name = "query-dnnf";
  version = "0.4.180625";

  src = fetchurl {
    url = "https://www.cril.univ-artois.fr/kc/ressources/query-dnnf-0.4.180625.zip";
    sha256 = "1fqvb90fbs1ayq37b813340qcpxqlv5mdzjgka81fj3qavfvds4s";
  };

  buildInputs = [ gmp ];
  nativeBuildInputs = [ unzip ];
}
