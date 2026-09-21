class LPCDetect(Detect):

    def __init__(self, nc=8, alpha_init=0.05, hidden_ratio=0.125, ch=()):
        super().__init__(nc=nc, ch=ch)

        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))

        self.lesion = nn.ModuleList()
        for c in ch:
            hidden = max(16, int(c * hidden_ratio))
            self.lesion.append(
                nn.Sequential(
                    Conv(c, hidden, 1, 1),
                    Conv(hidden, hidden, 3, 1, g=hidden),
                    nn.Conv2d(hidden, 1, kernel_size=1, stride=1, padding=0)
                )
            )

        for m in self.lesion:
            nn.init.zeros_(m[-1].weight)
            nn.init.zeros_(m[-1].bias)

    def forward(self, x):
        lesion_out = []

        for i in range(self.nl):
            xi = x[i]

            box_logits = self.cv2[i](xi)
            cls_logits = self.cv3[i](xi)

            lesion_logits = self.lesion[i](xi)
            lesion_out.append(lesion_logits)

            lesion_calib = torch.tanh(lesion_logits)
            alpha = torch.clamp(self.alpha, 0.0, 0.5)

            cls_logits = cls_logits + alpha * lesion_calib

            x[i] = torch.cat((box_logits, cls_logits), 1)

        if self.training:
            return x, lesion_out

        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)

        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (
                x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5)
            )
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor(
                [grid_w, grid_h, grid_w, grid_h],
                device=box.device
            ).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(
                self.dfl(box) * norm,
                self.anchors.unsqueeze(0) * norm[:, :2]
            )
        else:
            dbox = self.decode_bboxes(
                self.dfl(box),
                self.anchors.unsqueeze(0)
            ) * self.strides

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)
