package Bio::DB::BigWig;

use strict;
use warnings;
use Exporter 'import';

our @EXPORT_OK = qw(binMean);

sub binMean {
    return $_[0];
}

sub new {
    die "Bio::DB::BigWig shim was called, but LOFTEE should run with use_gerp_end_trunc:0";
}

1;
