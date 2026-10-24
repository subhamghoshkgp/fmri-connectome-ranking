% Run in MATLAB in the directory containing the original Dataset-1 MAT file.
% This reads only the named top-level variable, then exports functional data.
% Original filename can be edited; no original file is modified.
S = load('27_SCHZ_CTRL_dataset.mat', 'SC_FC_Connectomes');
FC = S.SC_FC_Connectomes.FC_correlation;
groups = {'ctrl', 'schz'};
out = cell(1,2);
for g = 1:2
    scales = FC.(groups{g});
    assert(iscell(scales), 'Unexpected layout: FC group must be a cell array.');
    found = 0;
    for k = 1:numel(scales)
        A = scales{k};
        if ndims(A)==3 && sum(size(A)==129)==2
            found = found+1;
            out{g} = A;
        end
    end
    assert(found==1, 'Expected one 129-ROI scale for each diagnosis.');
    assert(numel(out{g})==27*129*129, 'Expected 27 subjects per group.');
end
FC_ctrl_129 = out{1};
FC_schz_129 = out{2};
save('Dataset1_FC129_only.mat', 'FC_ctrl_129', 'FC_schz_129', '-v7');
disp('Created Dataset1_FC129_only.mat. Upload this smaller file.');
