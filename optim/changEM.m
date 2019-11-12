function [A] = changEM(A, newval, oldval)
% CHANGEM  
% [A] = changEM(A, newval, oldval)
%
% Substitute values in data array

%
%   A = CHANGEM(A,NEWVAL,OLDVAL) replaces all occurrences of NEWVAL(k) in A
%   with OLDVAL(k).  NEWVAL and OLDVAL must match in size.

% Copyright 2017 Kai Han.

%  Test that old and new value arrays have the same number of elements.
if numel(newval) ~= numel(oldval)
    error('Inconsistent sizes for old and new code inputs')
end

[ia, ib] = ismember(A,oldval);
ib = ib(ia);
A(ia) = newval(ib);

end