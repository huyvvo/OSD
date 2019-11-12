function [classes] = get_classes(imgset)
% GET_CLASSES
%
% [classes] = get_classes(imgset)

valid_sets = {'od', 'vocx', 'vocxss', 'vocxssfast', 'voc', ...
			  'vocxssquality'};

if strcmp(imgset(1:2), 'od')
	classes = {'airplane'; 'car'; 'horse'; 'mixed'};
elseif length(imgset) >= 4 & strcmp(imgset(1:4), 'vocx')
	classes = {'aeroplane_left'; 'aeroplane_right'; ...
	           'bicycle_left'; 'bicycle_right'; ...
	           'boat_left'; 'boat_right'; ...
	           'bus_left'; 'bus_right'; ...
	           'horse_left'; 'horse_right'; ...
	           'motorbike_left'; 'motorbike_right'; ...
	           'mixed' ...
	};
elseif strcmp(imgset(1:3), 'voc')
	classes = { ...
		  'aeroplane'; 'bicycle'; 'bird'; 'boat'; 'bottle'; 'bus'; ...
		  'car'; 'cat'; 'chair'; 'cow'; 'diningtable'; 'dog'; ...
		  'horse'; 'motorbike'; 'person'; 'pottedplant'; 'sheep'; ...
		  'sofa'; 'train'; 'tvmonitor' ; 'mixed' ...
	};
else 
	fprintf("'imgset' must be one of the sets");
	fprintf(' %s', valid_sets{:});
	fprintf('\n');
	error('Invalid value of "imgset"');
end

end