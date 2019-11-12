classdef LinearLRUpdater
	properties
		init_lr % initial learning rate
		min_lr % a lower bound for the learning rate
		max_iter % maximum number of iterations
	end

	methods
		function self = LinearLRUpdater(init_lr, min_lr, max_iter)
			self.init_lr = init_lr;
			self.min_lr = min_lr;
			self.max_iter = max_iter;
		end

		function lr = get_lr(self, iter)
			lr = max(self.min_lr, self.init_lr * (1 - (iter-1)/self.max_iter));
		end
	end
end