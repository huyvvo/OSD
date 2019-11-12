classdef DataLoader
  properties
    S   % matrix of cells of similarity
    files % if S is given as files or a cell array
    proposal_indices % 
  end

  methods
    function self = DataLoader(S, proposal_indices)
      self.S = S;
      % make sure that if S is given as files in a folder then
      % the folder exists
      if strcmp(class(self.S), 'char') || strcmp(class(self.S), 'string')
        assert(exist(self.S) == 7);
        self.files = true;
      else
        self.files = false;
      end
      if nargin == 1
        self.proposal_indices = [];
      else
        self.proposal_indices = proposal_indices;
      end
    end % function

    function res = get_S(self, i,j)
      if ~self.files

        if i < j
          res = self.S{i,j};
        elseif i > j
          res = transpose(self.S{j,i});
        else
          error('i and j must be different...');
        end

      else

        if i < j
          res = load(fullfile(self.S, sprintf('%d_%d.mat', i, j)));
          res = res.data;
        elseif i > j
          res = load(fullfile(self.S, sprintf('%d_%d.mat', j, i)));
          res = transpose(res.data);
        else
          error('i and j must be different...');
        end

      end

      if ~isempty(self.proposal_indices)
        res = res(self.proposal_indices{i}, self.proposal_indices{j});
      end

    end % function

    function obj = set.S(obj, S)
      obj.S = S;
      % make sure that if S is given as files in a folder then
      % the folder exists
      if strcmp(class(obj.S), 'char') | strcmp(class(obj.S), 'string')
        assert(exist(obj.S) == 7);
        obj.files = true;
      else
        obj.files = false;
      end
    end % function

    function obj = set.proposal_indices(obj, proposal_indices)
      obj.proposal_indices = proposal_indices;
    end % function

  end % method

end % class