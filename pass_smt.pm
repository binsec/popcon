## -*- mode: Perl -*-
##
## Copyright (c) 2012, 2015, 2016 The University of Utah
## All rights reserved.
##
## This file is distributed under the University of Illinois Open Source
## License.  See the file COPYING for details.

###############################################################################


# PERL5LIB=.:$PERL5LIB creduce --no-default-passes --timeout 110 --add-pass pass_balanced parens 1 --add-pass pass_smt true 2 --add-pass pass_smt false 3 --add-pass pass_smt first 3 --add-pass pass_smt second 4 --add-pass pass_smt third 5 --n 11 test.sh bug_reduce.smt2

package pass_smt;

use strict;
use warnings;

use Regexp::Common;
use re 'eval';

use creduce_regexes;
use creduce_utils;

sub check_prereqs () {
    return 1;
}

sub new ($$) {
    my $pos = 0;
    return \$pos;
}

sub advance ($$$) {
    (my $cfile, my $arg, my $state) = @_;
    my $pos = ${$state};
    $pos++;
    return \$pos;
}

sub remove_outside ($) {
    (my $str) = @_;
    substr($str,0,1) = "";
    substr($str,-1,1) = "";
    return $str;
}

# this function is idiotically stupid and slow but I spent a long time
# trying to get nested matches out of Perl's various utilities for
# matching balanced delimiters, with no success

sub transform ($$$) {
    (my $cfile, my $arg, my $state) = @_;

    my $pos = ${$state};
    my $prog = read_file ($cfile);

    while (1) {

	my $first = substr ($prog, 0, $pos);
	my $rest = substr ($prog, $pos);
	my $rest2 = $rest;

	my $context = substr($rest, 0, 30);
	# print "Now: $context\n";
	if (0) {
	} elsif ($arg eq "true") {
	    # replaces a boolean operator like (= ...) by true
	    if ($rest =~ /^\((=|distinct|bv[us][gl][et]|not|=>|and|or|xor)/) {
		$rest2 =~ s/^(?<all>($RE{balanced}{-parens=>'()'}))/true/s;
	    }
	} elsif ($arg eq "false") {
	    # replaces a boolean operator like (= ...) by true
	    if ($rest =~ /^\((=|distinct|bv[us][gl][et]|not|=>|and|or|xor)/) {
		$rest2 =~ s/^(?<all>($RE{balanced}{-parens=>'()'}))/false/s;
	    }
	} elsif ($arg eq "first") {
	    # replaces an operator like (foo a...) by its first argument a
	    if ($rest2 =~ /^(?<begin>\([^ ()]+ )(?<firstarg>(($RE{balanced}{-parens=>'()'})|[^ ()]+))/) {
		my $arg = $+{firstarg};
		# print "found operator $+{begin} with first argument $arg in $context\n";
		$rest2 =~ s/^(?<all>($RE{balanced}{-parens=>'()'}))//s;
		$rest2 = $arg . $rest2;
	    }
	} elsif ($arg eq "second") {
	    # replaces an operator like (foo a b...) by its second argument b
	    if ($rest2 =~ /^(?<begin>\([^ ()]+ )(?<firstarg>(($RE{balanced}{-parens=>'()'})|[^ ()]+)) (?<secondarg>(($RE{balanced}{-parens=>'()'})|[^ ()]+))/) {
		my $arg = $+{secondarg};
		# print "found operator $+{begin} with second argument $arg in $context\n";
		$rest2 =~ s/^(?<all>($RE{balanced}{-parens=>'()'}))//s;
		$rest2 = $arg . $rest2;
	    }
	} elsif ($arg eq "third") {
	    # replaces an operator like (foo a b...) by its third argument c
	    if ($rest2 =~ /^(?<begin>\([^ ()]+ )(?<firstarg>(($RE{balanced}{-parens=>'()'})|[^ ()]+)) (?<secondarg>(($RE{balanced}{-parens=>'()'})|[^ ()]+)) (?<thirdarg>(($RE{balanced}{-parens=>'()'})|[^ ()]+))/) {
		my $arg = $+{thirdarg};
		# print "found operator $+{begin} with third arg argument $arg in $context\n";
		$rest2 =~ s/^(?<all>($RE{balanced}{-parens=>'()'}))//s;
		$rest2 = $arg . $rest2;
	    }
	} else {
	    return ($ERROR, "unexpected argument");
	}
	if ($rest ne $rest2) {
	    write_file ($cfile, $first . $rest2);
	    return ($OK, \$pos);
	}
	$pos++;
	if ($pos > length($prog)) {
	    return ($STOP, \$pos);
	}
    }
}

1;
