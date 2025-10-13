package dynamo

import (
	"fmt"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/cloud/operator/api/v1alpha1"
	"github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

func TestVLLMBackend_UpdateContainer(t *testing.T) {
	backend := &VLLMBackend{}

	tests := []struct {
		name                string
		numberOfNodes       int32
		role                Role
		component           *v1alpha1.DynamoComponentDeploymentSharedSpec
		multinodeDeployer   MultinodeDeployer
		initialContainer    *corev1.Container
		expectedArgs        []string
		expectNotModified   bool // If true, container args should not change
		expectProbesRemoved bool // If true, probes should be nil
	}{
		{
			name:              "single node does not modify args",
			numberOfNodes:     1,
			role:              RoleMain,
			component:         &v1alpha1.DynamoComponentDeploymentSharedSpec{},
			multinodeDeployer: &GroveMultinodeDeployer{},
			initialContainer:  &corev1.Container{Args: []string{"python3", "-m", "dynamo.vllm"}},
			expectNotModified: true,
		},
		{
			name:                "multinode leader prepends ray start --head",
			numberOfNodes:       3,
			role:                RoleLeader,
			component:           &v1alpha1.DynamoComponentDeploymentSharedSpec{},
			multinodeDeployer:   &GroveMultinodeDeployer{},
			initialContainer:    &corev1.Container{Args: []string{"python3", "-m", "dynamo.vllm", "--model", "test", tensorParallelSizeFlag, "8"}, Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{corev1.ResourceName("nvidia.com/gpu"): resource.MustParse("4")}}},
			expectedArgs:        []string{fmt.Sprintf("ray start --head --port=%s && python3 -m dynamo.vllm --model test %s 8", VLLMPort, tensorParallelSizeFlag)},
			expectProbesRemoved: true,
		},
		{
			name:                "multinode worker replaces args with ray start --block",
			numberOfNodes:       3,
			role:                RoleWorker,
			component:           &v1alpha1.DynamoComponentDeploymentSharedSpec{},
			multinodeDeployer:   &GroveMultinodeDeployer{},
			initialContainer:    &corev1.Container{Args: []string{"python3", "-m", "dynamo.vllm", "--model", "test", tensorParallelSizeFlag, "8"}, Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{corev1.ResourceName("nvidia.com/gpu"): resource.MustParse("4")}}},
			expectedArgs:        []string{"ray start --address=$(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-test-service-ldr-0.$(GROVE_HEADLESS_SERVICE):6379 --block"},
			expectProbesRemoved: true,
		},
		{
			name:                "multinode worker with LWS deployment type",
			numberOfNodes:       2,
			role:                RoleWorker,
			component:           &v1alpha1.DynamoComponentDeploymentSharedSpec{},
			multinodeDeployer:   &LWSMultinodeDeployer{},
			initialContainer:    &corev1.Container{Args: []string{"python3", "-m", "dynamo.vllm", tensorParallelSizeFlag, "8"}, Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{corev1.ResourceName("nvidia.com/gpu"): resource.MustParse("4")}}},
			expectedArgs:        []string{"ray start --address=$(LWS_LEADER_ADDRESS):6379 --block"},
			expectProbesRemoved: true,
		},
		{
			name:              "multinode leader with no initial args",
			numberOfNodes:     2,
			role:              RoleLeader,
			component:         &v1alpha1.DynamoComponentDeploymentSharedSpec{},
			multinodeDeployer: &GroveMultinodeDeployer{},
			initialContainer:  &corev1.Container{Args: []string{}},
			expectNotModified: true, // Should not modify empty args
		},
		{
			name:              "multinode main role (non-leader/worker) does not modify args",
			numberOfNodes:     3,
			role:              RoleMain,
			component:         &v1alpha1.DynamoComponentDeploymentSharedSpec{},
			multinodeDeployer: &GroveMultinodeDeployer{},
			initialContainer:  &corev1.Container{Args: []string{"python3", "-m", "dynamo.frontend"}},
			expectNotModified: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)

			// Call UpdateContainer
			backend.UpdateContainer(tt.initialContainer, tt.numberOfNodes, tt.role, tt.component, "test-service", tt.multinodeDeployer)

			if tt.expectNotModified {
				// Args should not have changed
				g.Expect(tt.initialContainer.Args).To(gomega.Equal(tt.initialContainer.Args))
			} else if tt.expectedArgs != nil {
				// Check exact match
				g.Expect(tt.initialContainer.Args).To(gomega.Equal(tt.expectedArgs))
			}

			if tt.expectProbesRemoved {
				g.Expect(tt.initialContainer.LivenessProbe).To(gomega.BeNil())
				g.Expect(tt.initialContainer.ReadinessProbe).To(gomega.BeNil())
				g.Expect(tt.initialContainer.StartupProbe).To(gomega.BeNil())
			}
		})
	}
}

func TestVLLMBackend_UpdateContainer_UseAsCompilationCache(t *testing.T) {
	backend := &VLLMBackend{}

	tests := []struct {
		name                  string
		component             *v1alpha1.DynamoComponentDeploymentSharedSpec
		volumeMounts          []corev1.VolumeMount
		expectCacheEnvVar     bool
		expectCacheEnvVarName string
		expectCacheEnvVarVal  string
	}{
		{
			name: "VLLM backend with useAsCompilationCache volume mount",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				VolumeMounts: []v1alpha1.VolumeMount{
					{
						Name:                  "vllm-cache",
						MountPoint:            "/root/.cache/vllm",
						UseAsCompilationCache: true,
					},
				},
			},
			volumeMounts:          []corev1.VolumeMount{},
			expectCacheEnvVar:     true,
			expectCacheEnvVarName: "VLLM_CACHE_ROOT",
			expectCacheEnvVarVal:  "/root/.cache/vllm",
		},
		{
			name: "VLLM backend with useAsCompilationCache at custom mount point",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				VolumeMounts: []v1alpha1.VolumeMount{
					{
						Name:                  "custom-cache",
						MountPoint:            "/custom/cache/path",
						UseAsCompilationCache: true,
					},
				},
			},
			volumeMounts:          []corev1.VolumeMount{},
			expectCacheEnvVar:     true,
			expectCacheEnvVarName: "VLLM_CACHE_ROOT",
			expectCacheEnvVarVal:  "/custom/cache/path",
		},
		{
			name: "VLLM backend without useAsCompilationCache",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				VolumeMounts: []v1alpha1.VolumeMount{
					{
						Name:       "regular-volume",
						MountPoint: "/data",
					},
				},
			},
			volumeMounts:      []corev1.VolumeMount{},
			expectCacheEnvVar: false,
		},
		{
			name: "VLLM backend with no volume mounts",
			component: &v1alpha1.DynamoComponentDeploymentSharedSpec{
				VolumeMounts: nil,
			},
			volumeMounts:      []corev1.VolumeMount{},
			expectCacheEnvVar: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)

			// Create a container with initial state including volume mounts
			container := &corev1.Container{
				Env:          []corev1.EnvVar{},
				VolumeMounts: tt.volumeMounts,
			}

			// Call UpdateContainer
			backend.UpdateContainer(container, 1, RoleMain, tt.component, "test-service", &GroveMultinodeDeployer{})

			if tt.expectCacheEnvVar {
				// Check that the VLLM_CACHE_ROOT environment variable is set
				found := false
				for _, env := range container.Env {
					if env.Name == tt.expectCacheEnvVarName {
						found = true
						g.Expect(env.Value).To(gomega.Equal(tt.expectCacheEnvVarVal))
						break
					}
				}
				if !found {
					t.Errorf("Expected environment variable %s not found in container", tt.expectCacheEnvVarName)
				}
			} else {
				// Check that no cache environment variable is set
				for _, env := range container.Env {
					if env.Name == "VLLM_CACHE_ROOT" {
						t.Errorf("Unexpected environment variable VLLM_CACHE_ROOT found: %s", env.Value)
					}
				}
			}
		})
	}
}

func TestUpdateVLLMMultinodeArgs(t *testing.T) {
	tests := []struct {
		name              string
		role              Role
		multinodeDeployer MultinodeDeployer
		initialContainer  *corev1.Container
		expectedArgs      []string
		expectNotModified bool
	}{
		{
			name:              "leader prepends ray start --head",
			role:              RoleLeader,
			multinodeDeployer: &GroveMultinodeDeployer{},
			initialContainer:  &corev1.Container{Args: []string{"python3", "-m", "dynamo.vllm", tensorParallelSizeFlag, "16"}, Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{corev1.ResourceName("nvidia.com/gpu"): resource.MustParse("8")}}},
			expectedArgs:      []string{fmt.Sprintf("ray start --head --port=%s && python3 -m dynamo.vllm %s 16", VLLMPort, tensorParallelSizeFlag)},
		},
		{
			name:              "leader prepends distributed data parallel flags",
			role:              RoleLeader,
			multinodeDeployer: &GroveMultinodeDeployer{},
			initialContainer:  &corev1.Container{Command: []string{"python3"}, Args: []string{"-m", "dynamo.vllm", dataParallelSizeFlag, "16"}, Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{corev1.ResourceName("nvidia.com/gpu"): resource.MustParse("8")}}},
			expectedArgs:      []string{fmt.Sprintf("exec python3 -m dynamo.vllm %s 16 --data-parallel-address $(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-test-service-ldr-0.$(GROVE_HEADLESS_SERVICE) --data-parallel-size-local 8 --data-parallel-rpc-port 13445 --data-parallel-start-rank 0", dataParallelSizeFlag)},
		},
		{
			name:              "leader with empty args does not modify",
			role:              RoleLeader,
			multinodeDeployer: &GroveMultinodeDeployer{},
			initialContainer:  &corev1.Container{Args: []string{}},
			expectNotModified: true,
		},
		{
			name:              "worker with ray distributed launch Grove",
			role:              RoleWorker,
			multinodeDeployer: &GroveMultinodeDeployer{},
			initialContainer:  &corev1.Container{Args: []string{"python3", "-m", "dynamo.vllm", tensorParallelSizeFlag, "16"}, Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{corev1.ResourceName("nvidia.com/gpu"): resource.MustParse("8")}}},
			expectedArgs:      []string{"ray start --address=$(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-test-service-ldr-0.$(GROVE_HEADLESS_SERVICE):6379 --block"},
		},
		{
			name:              "worker with data parallel launch Grove",
			role:              RoleWorker,
			multinodeDeployer: &GroveMultinodeDeployer{},
			initialContainer:  &corev1.Container{Command: []string{"python3"}, Args: []string{"-m", "dynamo.vllm", dataParallelSizeFlag, "16"}, Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{corev1.ResourceName("nvidia.com/gpu"): resource.MustParse("8")}}},
			expectedArgs:      []string{fmt.Sprintf("exec python3 -m dynamo.vllm %s 16 --data-parallel-address $(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-test-service-ldr-0.$(GROVE_HEADLESS_SERVICE) --data-parallel-size-local 8 --data-parallel-rpc-port 13445 --data-parallel-start-rank $(( 8 * $((GROVE_PCLQ_POD_INDEX + 1)) ))", dataParallelSizeFlag)},
		},
		{
			name:              "worker with data parallel launch Grove, tp > 1",
			role:              RoleWorker,
			multinodeDeployer: &GroveMultinodeDeployer{},
			initialContainer:  &corev1.Container{Command: []string{"python3"}, Args: []string{"-m", "dynamo.vllm", dataParallelSizeFlag, "8", tensorParallelSizeFlag, "2"}, Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{corev1.ResourceName("nvidia.com/gpu"): resource.MustParse("8")}}},
			expectedArgs:      []string{fmt.Sprintf("exec python3 -m dynamo.vllm %s 8 %s 2 --data-parallel-address $(GROVE_PCSG_NAME)-$(GROVE_PCSG_INDEX)-test-service-ldr-0.$(GROVE_HEADLESS_SERVICE) --data-parallel-size-local 4 --data-parallel-rpc-port 13445 --data-parallel-start-rank $(( 4 * $((GROVE_PCLQ_POD_INDEX + 1)) ))", dataParallelSizeFlag, tensorParallelSizeFlag)},
		},
		{
			name:              "worker with ray distributed launch LWS",
			role:              RoleWorker,
			multinodeDeployer: &LWSMultinodeDeployer{},
			initialContainer:  &corev1.Container{Args: []string{"python3", "-m", "dynamo.vllm", tensorParallelSizeFlag, "16"}, Resources: corev1.ResourceRequirements{Limits: corev1.ResourceList{corev1.ResourceName("nvidia.com/gpu"): resource.MustParse("8")}}},
			expectedArgs:      []string{"ray start --address=$(LWS_LEADER_ADDRESS):6379 --block"},
		},
		{
			name:              "main role does not modify args",
			role:              RoleMain,
			multinodeDeployer: &GroveMultinodeDeployer{},
			initialContainer:  &corev1.Container{Args: []string{"python3", "-m", "dynamo.frontend"}},
			expectNotModified: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)

			initialContainerArgs := append([]string{}, tt.initialContainer.Args...)
			// Call updateVLLMMultinodeArgs
			updateVLLMMultinodeArgs(tt.initialContainer, tt.role, "test-service", tt.multinodeDeployer)

			if tt.expectNotModified {
				// Args should not have changed
				g.Expect(tt.initialContainer.Args).To(gomega.Equal(initialContainerArgs))
			} else if tt.expectedArgs != nil {
				// Check exact match
				g.Expect(tt.initialContainer.Args).To(gomega.Equal(tt.expectedArgs))
			}
		})
	}
}
